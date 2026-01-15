# Memory Issue Fix - engine.py Standard Mode

**Issue Date:** 2026-01-14
**File:** `src/analysis/engine.py`
**Problem:** Memory exhaustion in Standard (Per-Strategy) mode

---

## Problem Analysis

### Root Cause

The issue is **NOT** primarily about too many workers. The real problem is **inefficient data filtering** in the main process.

**Location:** `engine.py`, lines 800, 946, 1093, 1098

```python
# This line runs for EVERY strategy (6144 times!)
strat_data = self.data[self.data["No."] == strategy_id]
```

### Why This Is Inefficient

1. `self.data` has **1,277,952 rows**
2. `self.data["No."]` creates a Series copy (1.2M elements)
3. `== strategy_id` creates a **boolean array** (1.2M bools = ~1.22 MB)
4. The filtering `self.data[mask]` creates another DataFrame slice

**For 6144 strategies:** This means ~7.5 GB of temporary boolean arrays!

### Log Evidence

```
22:23:40 | ERROR | numpy._core._exceptions._ArrayMemoryError:
Unable to allocate 1.22 MiB for an array with shape (1277952,) and data type bool
```

The system couldn't allocate **1.22 MiB** despite having 64GB RAM because:
- 19 workers were consuming memory
- Main process was repeatedly creating temporary arrays
- Memory became fragmented

---

## Recommended Fix

### Solution 1: Pre-grouping (Best Performance)

Create a dictionary of strategy data **once** before the loop:

```python
# Before the parallel loop (around line 785):
strategy_groups = {
    strat_id: group.sort_values("Date")
    for strat_id, group in self.data.groupby("No.")
}

# In the loop (replace line 946):
strat_data = strategy_groups[strategy_id]  # O(1) lookup!
```

**Benefits:**
- O(n) grouping happens once
- Each lookup is O(1)
- No repeated boolean array creation
- Memory usage is predictable

### Where to Apply

Apply this fix in **4 locations** in `engine.py`:

1. **Line 800** (single-worker mode):
   ```python
   strat_data = self.data[self.data["No."] == strategy_id].sort_values("Date")
   ```

2. **Line 946** (parallel mode - task submission):
   ```python
   strat_data = self.data[self.data["No."] == strategy_id]
   ```

3. **Line 1093** (post-processing - best strategy):
   ```python
   best_strat_data = self.data[self.data["No."] == best_strat_id].copy()
   ```

4. **Line 1098** (post-processing - all strategies):
   ```python
   strat_data = self.data[self.data["No."] == strategy_id].copy()
   ```

---

## Implementation Guide

### Step 1: Add Pre-grouping

After line 786 (`results = []`), add:

```python
# PRE-GROUPING: Create strategy lookup dictionary ONCE
# This eliminates repeated O(n) filtering in the main loop
logger.debug("Pre-grouping data by strategy ID...")
strategy_groups = {}
for strat_id, group in self.data.groupby("No."):
    # Pre-sort and drop unnecessary columns
    sorted_group = group.sort_values("Date")
    exclude_cols = ["Start", "1st Candle", "Shift", "Position", "Param. Sum", "SourceFile"]
    cols_to_drop = [c for c in exclude_cols if c in sorted_group.columns]
    if cols_to_drop:
        sorted_group = sorted_group.drop(columns=cols_to_drop)
    strategy_groups[strat_id] = sorted_group
logger.debug("Pre-grouping complete: %d strategies", len(strategy_groups))
```

### Step 2: Replace Filtering Calls

**Single-worker mode (line 800):**
```python
# OLD:
strat_data = self.data[self.data["No."] == strategy_id].sort_values("Date")

# NEW:
strat_data = strategy_groups[strategy_id]
```

**Parallel mode (line 946):**
```python
# OLD:
strat_data = self.data[self.data["No."] == strategy_id]
strat_data = strat_data.sort_values("Date")
exclude_cols = [...]
cols_to_drop = [c for c in exclude_cols if c in strat_data.columns]
if cols_to_drop:
    strat_data = strat_data.drop(columns=cols_to_drop)

# NEW:
strat_data = strategy_groups[strategy_id]
```

**Post-processing (lines 1093, 1098):**
```python
# OLD:
best_strat_data = self.data[self.data["No."] == best_strat_id].copy()
...
strat_data = self.data[self.data["No."] == strategy_id].copy()

# NEW:
best_strat_data = strategy_groups[best_strat_id].copy()
...
strat_data = strategy_groups[strategy_id].copy()
```

---

## Alternative Solutions

### Solution 2: Index-based Lookup

```python
# Set index once
if "No." not in self.data.index.names:
    indexed_data = self.data.set_index("No.", drop=False)

# Fast lookup
strat_data = indexed_data.loc[[strategy_id]]
```

### Solution 3: Reduce Buffer Size

Less impactful but helps:

```python
# Current (line 937):
buffer_size = optimal_workers * 2  # 19 * 2 = 38 tasks

# Reduced:
buffer_size = max(4, optimal_workers // 2)  # ~10 tasks
```

---

## Expected Results

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Memory per strategy lookup | ~1.22 MB temp | ~0 bytes temp |
| Total temp allocations | ~7.5 GB | ~0 GB |
| Memory stability | Fragmented | Stable |
| Failure rate | High (OOM) | Low |

---

## Testing

After applying the fix:

1. Run the auto-execution with the same 9 KNN configurations
2. Monitor RAM usage (should stay stable)
3. Verify all 9 runs complete successfully
4. Check results match previous successful run

---

*Fix designed: 2026-01-14*
*Analyst: Claude Code AI*
