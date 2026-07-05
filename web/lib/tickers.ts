/** Display metadata for the serving universe (25 NIFTY-50 constituents). */

export interface TickerMeta {
  name: string;
  sector: string;
}

export const TICKER_META: Record<string, TickerMeta> = {
  "RELIANCE.NS": { name: "Reliance Industries", sector: "Energy" },
  "TCS.NS": { name: "Tata Consultancy Services", sector: "IT" },
  "HDFCBANK.NS": { name: "HDFC Bank", sector: "Banking" },
  "ICICIBANK.NS": { name: "ICICI Bank", sector: "Banking" },
  "INFY.NS": { name: "Infosys", sector: "IT" },
  "HINDUNILVR.NS": { name: "Hindustan Unilever", sector: "FMCG" },
  "ITC.NS": { name: "ITC", sector: "FMCG" },
  "SBIN.NS": { name: "State Bank of India", sector: "Banking" },
  "BHARTIARTL.NS": { name: "Bharti Airtel", sector: "Telecom" },
  "KOTAKBANK.NS": { name: "Kotak Mahindra Bank", sector: "Banking" },
  "LT.NS": { name: "Larsen & Toubro", sector: "Infra" },
  "AXISBANK.NS": { name: "Axis Bank", sector: "Banking" },
  "BAJFINANCE.NS": { name: "Bajaj Finance", sector: "NBFC" },
  "ASIANPAINT.NS": { name: "Asian Paints", sector: "Consumer" },
  "MARUTI.NS": { name: "Maruti Suzuki", sector: "Auto" },
  "HCLTECH.NS": { name: "HCL Technologies", sector: "IT" },
  "SUNPHARMA.NS": { name: "Sun Pharmaceutical", sector: "Pharma" },
  "TITAN.NS": { name: "Titan Company", sector: "Consumer" },
  "ULTRACEMCO.NS": { name: "UltraTech Cement", sector: "Cement" },
  "WIPRO.NS": { name: "Wipro", sector: "IT" },
  "NESTLEIND.NS": { name: "Nestlé India", sector: "FMCG" },
  "M&M.NS": { name: "Mahindra & Mahindra", sector: "Auto" },
  "TATASTEEL.NS": { name: "Tata Steel", sector: "Metals" },
  "POWERGRID.NS": { name: "Power Grid Corp", sector: "Utilities" },
  "NTPC.NS": { name: "NTPC", sector: "Utilities" },
};

export function metaOf(ticker: string): TickerMeta {
  return TICKER_META[ticker] ?? { name: ticker.replace(".NS", ""), sector: "—" };
}

/** "RELIANCE.NS" -> "RELIANCE" for compact display. */
export function shortSymbol(ticker: string): string {
  return ticker.replace(".NS", "");
}
