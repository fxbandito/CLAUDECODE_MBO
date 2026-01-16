"""
MT5 Report Generator - Parses MT5 HTML reports and generates modern Dark Theme reports.
"""
import os
import re
import base64
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TestResults:
    """Data class for backtest results."""
    # Settings
    expert: str = ""
    symbol: str = ""
    period: str = ""
    company: str = ""
    currency: str = ""
    initial_deposit: str = ""
    leverage: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)

    # Quality & History
    history_quality: str = ""
    bars: str = ""
    ticks: str = ""
    symbols: str = ""

    # Results - Main
    net_profit: str = ""
    total_profit: str = ""
    total_loss: str = ""
    profit_factor: str = ""
    recovery_factor: str = ""
    sharpe_ratio: str = ""
    expected_payoff: str = ""
    margin_level: str = ""
    z_score: str = ""

    # AHPR/GHPR/LR
    ahpr: str = ""
    ghpr: str = ""
    lr_correlation: str = ""
    lr_standard_error: str = ""
    ontester_result: str = ""

    # Drawdown
    balance_dd_absolute: str = ""
    balance_dd_max: str = ""
    balance_dd_relative: str = ""
    equity_dd_absolute: str = ""
    equity_dd_max: str = ""
    equity_dd_relative: str = ""

    # Trades
    total_trades: str = ""
    total_deals: str = ""
    short_trades: str = ""
    long_trades: str = ""
    profit_trades: str = ""
    loss_trades: str = ""
    largest_profit: str = ""
    largest_loss: str = ""
    avg_profit: str = ""
    avg_loss: str = ""
    max_consecutive_wins: str = ""
    max_consecutive_losses: str = ""
    max_consecutive_profit: str = ""
    max_consecutive_loss: str = ""
    avg_consecutive_wins: str = ""
    avg_consecutive_losses: str = ""

    # MFE/MAE Correlation
    correlation_profits_mfe: str = ""
    correlation_profits_mae: str = ""
    correlation_mfe_mae: str = ""

    # Holding Time
    min_holding_time: str = ""
    max_holding_time: str = ""
    avg_holding_time: str = ""

    # Images (base64 encoded)
    equity_chart: str = ""
    histogram_chart: str = ""
    mfemae_chart: str = ""
    holding_chart: str = ""


class MT5ReportParser:
    """Parser for MT5 HTML reports."""

    def __init__(self, path: str):
        self.report_path = path
        self.report_dir = os.path.dirname(path)
        self.report_name = os.path.splitext(os.path.basename(path))[0]

    def parse(self) -> TestResults:
        """Parse the MT5 HTML report and return TestResults."""
        results = TestResults()

        try:
            with open(self.report_path, 'r', encoding='utf-16') as f:
                content = f.read()
        except UnicodeError:
            with open(self.report_path, 'r', encoding='utf-8') as f:
                content = f.read()

        content = self._normalize_content(content)

        # Parse settings - Hungarian MT5 labels
        results.expert = self._extract_td_value(content, r'Expert\s*:')
        results.symbol = self._extract_td_value(content, r'Szimb[oó]lum\s*:')
        if not results.symbol:
            results.symbol = self._extract_td_value(content, r'Symbol\s*:')

        results.period = self._extract_td_value(content, r'Peri[oó]dus\s*:')
        if not results.period:
            results.period = self._extract_td_value(content, r'Period\s*:')

        results.company = self._extract_td_value(content, r'C[eé]g\s*:')
        results.currency = self._extract_td_value(content, r'P[eé]nznem\s*:')
        results.initial_deposit = self._extract_td_value(content, r'Kezdeti t[oő]ke\s*:')
        results.leverage = self._extract_td_value(content, r'T[oő]ke[aá]tt[eé]t\s*:')

        # Parse inputs
        input_pattern = r'<b>([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^<]+)</b>'
        for match in re.finditer(input_pattern, content):
            key, value = match.groups()
            results.inputs[key.strip()] = value.strip()

        # History quality
        results.history_quality = self._extract_td_value(content, r'El[oő]zm[eé]nyek min[oő]s[eé]ge\s*:')
        results.bars = self._extract_td_value(content, r'Oszlopok\s*:')
        results.ticks = self._extract_td_value(content, r'Tickek\s*:')
        results.symbols = self._extract_td_value(content, r'Szimb[oó]lumok\s*:')

        # Main results
        results.net_profit = self._extract_td_value(content, r'Nett[oó] nyeres[eé]g [oö]sszesen\s*:')
        results.total_profit = self._extract_td_value(content, r'[ÖO]sszes profit\s*:')
        results.total_loss = self._extract_td_value(content, r'[ÖO]sszes vesztes[eé]g\s*:')
        results.profit_factor = self._extract_td_value(content, r'Profit faktor\s*:')
        results.recovery_factor = self._extract_td_value(content, r'Meg[uú]jul[aá]si faktor\s*:')
        results.sharpe_ratio = self._extract_td_value(content, r'Sharpe ar[aá]ny\s*:')
        results.expected_payoff = self._extract_td_value(content, r'V[aá]rt Eredm[eé]ny\s*:')
        results.margin_level = self._extract_td_value(content, r'Margin szint\s*:')
        results.z_score = self._extract_td_value(content, r'Z-pont\s*:')

        # AHPR/GHPR
        results.ahpr = self._extract_td_value(content, r'AHPR\s*:')
        results.ghpr = self._extract_td_value(content, r'GHPR\s*:')
        results.lr_correlation = self._extract_td_value(content, r'LR korrel[aá]ci[oó]\s*:')
        results.lr_standard_error = self._extract_td_value(content, r'LR Standard Hiba\s*:')
        results.ontester_result = self._extract_td_value(content, r'OnTester eredm[eé]ny\s*:')

        # Drawdown
        results.balance_dd_absolute = self._extract_td_value(content, r'Egyenleg drawdown - abszol[uú]t\s*:')
        results.balance_dd_max = self._extract_td_value(content, r'Egyenleg drawdown - maxim[aá]lis\s*:')
        results.balance_dd_relative = self._extract_td_value(content, r'Egyenleg drawdown - relat[ií]v\s*:')
        results.equity_dd_absolute = self._extract_td_value(content, r'R[eé]szv[eé]ny cs[oö]kkent[eé]s abszol[uú]t\s*:')
        results.equity_dd_max = self._extract_td_value(content, r'R[eé]szv[eé]ny cs[oö]kkent[eé]s maxim[aá]lis\s*:')
        results.equity_dd_relative = self._extract_td_value(content, r'R[eé]szv[eé]ny cs[oö]kkent[eé]s relat[ií]v\s*:')

        # Trades
        results.total_trades = self._extract_td_value(content, r'[ÜU]gyletek [oö]sszesen\s*:', occurrence=0)
        results.total_deals = self._extract_td_value(content, r'[ÜU]gyletek [oö]sszesen\s*:', occurrence=1)
        results.short_trades = self._extract_td_value(content, r'Short [uü]gyletek[^:]*:')
        results.long_trades = self._extract_td_value(content, r'V[eé]teli [uü]gyletek[^:]*:')
        results.profit_trades = self._extract_td_value(content, r'Nyeres[eé]ges [uü]gyletek[^:]*:')
        results.loss_trades = self._extract_td_value(content, r'Vesztes[eé]ges [uü]gyletek[^:]*:')
        results.largest_profit = self._extract_td_value(content, r'Legnagyobb nyeres[eé]ges keresked[eé]s\s*:')
        results.largest_loss = self._extract_td_value(content, r'Legnagyobb vesztes[eé]ges keresked[eé]s\s*:')
        results.avg_profit = self._extract_td_value(content, r'[ÁA]tlag nyeres[eé]ges keresked[eé]s\s*:')
        results.avg_loss = self._extract_td_value(content, r'[ÁA]tlag vesztes[eé]ges keresked[eé]s\s*:')
        results.max_consecutive_wins = self._extract_td_value(content, r'Maximum\s+egym[aá]s ut[aá]ni nyeres[eé]g[^:]*:')
        results.max_consecutive_losses = self._extract_td_value(content, r'Maximum\s+egym[aá]s ut[aá]ni vesztes[eé]g[^:]*:')
        results.max_consecutive_profit = self._extract_td_value(content, r'Maxim[aá]lis egym[aá]s ut[aá]ni nyeres[eé]g[^:]*:')
        results.max_consecutive_loss = self._extract_td_value(content, r'Maxim[aá]lis egym[aá]s ut[aá]ni vesztes[eé]g[^:]*:')
        results.avg_consecutive_wins = self._extract_td_value(content, r'[ÁA]tlag egym[aá]st k[oö]vet[oő] nyeres[eé]gek\s*:')
        results.avg_consecutive_losses = self._extract_td_value(content, r'[ÁA]tlag egym[aá]st k[oö]vet[oő] vesztes[eé]gek\s*:')

        # MFE/MAE Correlation
        results.correlation_profits_mfe = self._extract_td_value(content, r'Correlation\s*\(Profits,\s*MFE\)\s*:')
        results.correlation_profits_mae = self._extract_td_value(content, r'Correlation\s*\(Profits,\s*MAE\)\s*:')
        results.correlation_mfe_mae = self._extract_td_value(content, r'Correlation\s*\(MFE,\s*MAE\)\s*:')

        # Holding Time
        results.min_holding_time = self._extract_td_value(content, r'Poz[ií]ci[oó] minim[aá]lis tart[aá]si ideje\s*:')
        results.max_holding_time = self._extract_td_value(content, r'Poz[ií]ci[oó] maximum tart[aá]si ideje\s*:')
        results.avg_holding_time = self._extract_td_value(content, r'[ÁA]tlagos poz[ií]ci[oó]tart[aá]si id[oő]\s*:')

        # Load images
        results.equity_chart = self._load_image(f"{self.report_name}.png")
        results.histogram_chart = self._load_image(f"{self.report_name}-hst.png")
        results.mfemae_chart = self._load_image(f"{self.report_name}-mfemae.png")
        results.holding_chart = self._load_image(f"{self.report_name}-holding.png")

        return results

    def _normalize_content(self, content: str) -> str:
        """Normalize UTF-16 content."""
        content = content.replace('\x00', '')
        content = re.sub(r' +', ' ', content)
        return content

    def _extract_td_value(self, content: str, label_pattern: str, occurrence: int = 0) -> str:
        """Extract value from MT5 HTML table structure."""
        pattern = label_pattern + r'</td>\s*<td[^>]*>\s*<b>([^<]*)</b>'
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        if matches and len(matches) > occurrence:
            return matches[occurrence].strip()
        return ""

    def _load_image(self, filename: str) -> str:
        """Load image and convert to base64."""
        image_path = os.path.join(self.report_dir, filename)
        if os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except (OSError, IOError):
                pass
        return ""


class DarkThemeReportGenerator:
    """Generates modern Dark Theme HTML reports."""

    DARK_THEME_CSS = """
    :root {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-card: #0f3460;
        --text-primary: #eaeaea;
        --text-secondary: #a0a0a0;
        --accent-green: #00d26a;
        --accent-red: #ff6b6b;
        --accent-blue: #4dabf7;
        --accent-gold: #ffd93d;
        --border-color: #2a2a4a;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
        min-height: 100vh;
        padding: 20px;
    }

    .container { max-width: 1200px; margin: 0 auto; }

    .header {
        text-align: center;
        padding: 30px 0;
        border-bottom: 2px solid var(--border-color);
        margin-bottom: 30px;
    }

    .header h1 { font-size: 2.5em; color: var(--accent-blue); margin-bottom: 10px; }
    .header .subtitle { color: var(--text-secondary); font-size: 1.2em; }

    .card {
        background: var(--bg-card);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid var(--border-color);
    }

    .card-title {
        font-size: 1.4em;
        color: var(--accent-gold);
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-color);
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 15px;
    }

    .stat-item {
        display: flex;
        justify-content: space-between;
        padding: 12px 15px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stat-item:hover { background: rgba(255, 255, 255, 0.1); transform: translateX(5px); }
    .stat-label { color: var(--text-secondary); }
    .stat-value { font-weight: bold; color: var(--text-primary); }
    .stat-value.positive { color: var(--accent-green); }
    .stat-value.negative { color: var(--accent-red); }

    .key-metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-bottom: 30px;
    }

    .metric-box {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-secondary));
        border-radius: 12px;
        padding: 15px 20px;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: transform 0.3s ease;
    }

    .metric-box:hover { transform: translateY(-3px); }
    .metric-value { font-size: 1.4em; font-weight: bold; margin-bottom: 5px; }
    .metric-label { color: var(--text-secondary); font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }

    .grid-2x2, .grid-3x2 { display: grid; gap: 15px; }
    .grid-2x2 { grid-template-columns: repeat(2, 1fr); }
    .grid-3x2 { grid-template-columns: repeat(3, 1fr); }

    .chart-container { text-align: center; margin: 20px 0; }
    .chart-container img { max-width: 100%; height: auto; border-radius: 10px; border: 1px solid var(--border-color); }
    .chart-title { color: var(--text-secondary); margin-bottom: 15px; font-size: 1.1em; }

    .inputs-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
    .input-item { background: rgba(255, 255, 255, 0.03); padding: 10px 15px; border-radius: 5px; font-family: 'Consolas', monospace; font-size: 0.9em; }
    .input-key { color: var(--accent-blue); }
    .input-value { color: var(--accent-gold); }

    .footer { text-align: center; padding: 20px; color: var(--text-secondary); font-size: 0.9em; border-top: 1px solid var(--border-color); margin-top: 30px; }

    @media (max-width: 768px) {
        .header h1 { font-size: 1.8em; }
        .key-metrics, .grid-3x2 { grid-template-columns: repeat(2, 1fr); }
        .metric-value { font-size: 1.2em; }
    }
    """

    def __init__(self, results: TestResults):
        self.results = results

    def generate(self, output_path: str) -> str:
        """Generate the Dark Theme HTML report."""
        html = self._build_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path

    def _build_html(self) -> str:
        """Build the complete HTML document."""
        r = self.results
        profit_class = "positive" if not r.net_profit.startswith('-') else "negative"
        pf_class = "positive" if self._parse_number(r.profit_factor) >= 1 else "negative"

        html = f"""<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MT5 Backtest Report - {r.expert}</title>
    <style>{self.DARK_THEME_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Strategy Tester Report</h1>
            <div class="subtitle">{r.expert} | {r.symbol} | {r.period}</div>
        </div>

        <div class="key-metrics">
            <div class="metric-box">
                <div class="metric-value {profit_class}">{r.net_profit}</div>
                <div class="metric-label">Net Profit</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {pf_class}">{r.profit_factor}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{r.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-box">
                <div class="metric-value negative">{r.balance_dd_max}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{r.sharpe_ratio}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{r.recovery_factor}</div>
                <div class="metric-label">Recovery Factor</div>
            </div>
        </div>

        {self._build_chart_section("Equity Curve", r.equity_chart)}
        {self._build_inputs_section()}

        <div class="card">
            <div class="card-title">Settings</div>
            <div class="grid-2x2">
                {self._stat_item("Expert Advisor", r.expert)}
                {self._stat_item("Symbol", r.symbol)}
                {self._stat_item("Period", r.period)}
                {self._stat_item("Initial Deposit", f"{r.initial_deposit} {r.currency}")}
                {self._stat_item("Company", r.company)}
                {self._stat_item("Leverage", r.leverage)}
            </div>
        </div>

        {self._build_history_section()}

        <div class="card">
            <div class="card-title">Results</div>
            <div class="stats-grid">
                {self._stat_item("Net Profit", r.net_profit, profit_class)}
                {self._stat_item("Total Profit", r.total_profit, "positive")}
                {self._stat_item("Total Loss", r.total_loss, "negative")}
                {self._stat_item("Profit Factor", r.profit_factor, pf_class)}
                {self._stat_item("Expected Payoff", r.expected_payoff)}
                {self._stat_item("Recovery Factor", r.recovery_factor)}
                {self._stat_item("Sharpe Ratio", r.sharpe_ratio)}
                {self._stat_item("Z-Score", r.z_score)}
            </div>
        </div>

        <div class="card">
            <div class="card-title">Drawdown</div>
            <div class="grid-3x2">
                {self._stat_item("Balance DD (Absolute)", r.balance_dd_absolute, "negative")}
                {self._stat_item("Balance DD (Maximum)", r.balance_dd_max, "negative")}
                {self._stat_item("Balance DD (Relative)", r.balance_dd_relative, "negative")}
                {self._stat_item("Equity DD (Absolute)", r.equity_dd_absolute, "negative")}
                {self._stat_item("Equity DD (Maximum)", r.equity_dd_max, "negative")}
                {self._stat_item("Equity DD (Relative)", r.equity_dd_relative, "negative")}
            </div>
        </div>

        {self._build_chart_section("Trade Distribution", r.histogram_chart)}

        <div class="card">
            <div class="card-title">Trades Statistics</div>
            <div class="stats-grid">
                {self._stat_item("Total Trades", r.total_trades)}
                {self._stat_item("Total Deals", r.total_deals)}
                {self._stat_item("Short Trades (Win %)", r.short_trades)}
                {self._stat_item("Long Trades (Win %)", r.long_trades)}
                {self._stat_item("Profit Trades", r.profit_trades, "positive")}
                {self._stat_item("Loss Trades", r.loss_trades, "negative")}
                {self._stat_item("Largest Profit", r.largest_profit, "positive")}
                {self._stat_item("Largest Loss", r.largest_loss, "negative")}
                {self._stat_item("Average Profit", r.avg_profit, "positive")}
                {self._stat_item("Average Loss", r.avg_loss, "negative")}
            </div>
        </div>

        {self._build_chart_section("MFE/MAE Analysis", r.mfemae_chart)}
        {self._build_chart_section("Position Holding Time", r.holding_chart)}

        <div class="footer">Generated by MT5 Backtester | Dark Theme Report</div>
    </div>
</body>
</html>"""
        return html

    def _stat_item(self, label: str, value: str, css_class: str = "") -> str:
        """Build a stat item HTML."""
        class_str = f' class="stat-value {css_class}"' if css_class else 'class="stat-value"'
        return f"""<div class="stat-item">
            <span class="stat-label">{label}</span>
            <span {class_str}>{value}</span>
        </div>"""

    def _build_chart_section(self, title: str, image_data: str) -> str:
        """Build a chart section with base64 image."""
        if not image_data:
            return ""
        return f"""<div class="card">
            <div class="chart-container">
                <div class="chart-title">{title}</div>
                <img src="data:image/png;base64,{image_data}" alt="{title}">
            </div>
        </div>"""

    def _build_inputs_section(self) -> str:
        """Build the EA inputs section."""
        if not self.results.inputs:
            return ""
        inputs_html = ""
        for key, value in self.results.inputs.items():
            inputs_html += f'<div class="input-item"><span class="input-key">{key}</span> = <span class="input-value">{value}</span></div>'
        return f"""<div class="card">
            <div class="card-title">EA Inputs</div>
            <div class="inputs-grid">{inputs_html}</div>
        </div>"""

    def _build_history_section(self) -> str:
        """Build the history quality section."""
        r = self.results
        if not r.history_quality:
            return ""
        return f"""<div class="card">
            <div class="card-title">History Quality</div>
            <div class="grid-2x2">
                {self._stat_item("Quality", r.history_quality)}
                {self._stat_item("Bars", r.bars)}
                {self._stat_item("Ticks", r.ticks)}
                {self._stat_item("Symbols", r.symbols)}
            </div>
        </div>"""

    def _parse_number(self, value: str) -> float:
        """Parse a number from string."""
        try:
            clean = value.replace(' ', '').replace(',', '.')
            clean = re.sub(r'[^\d.\-]', '', clean)
            return float(clean) if clean else 0
        except (ValueError, AttributeError):
            return 0


def generate_report(mt5_report_path: str, output_path: Optional[str] = None) -> str:
    """Generate a Dark Theme report from an MT5 HTML report."""
    if output_path is None:
        base = os.path.splitext(mt5_report_path)[0]
        output_path = f"{base}_dark.html"

    parser = MT5ReportParser(mt5_report_path)
    results = parser.parse()
    generator = DarkThemeReportGenerator(results)
    return generator.generate(output_path)
