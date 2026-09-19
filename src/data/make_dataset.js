const fs = require('fs');
const path = require('path');
const readline = require('readline');
const XLSX = require('xlsx');

// Relative paths inside repo or fallback to user Downloads
const defaultDownloads = 'C:\\Users\\novag\\Downloads';
const projectRoot = path.resolve(__dirname, '..', '..');

const underpricedPath = path.join(defaultDownloads, 'underpriced_SME');
const mergedExcelPath = path.join(defaultDownloads, 'final_merged_sme_data (1).xlsx');
const tradingCsvPath = path.join(defaultDownloads, '103138_1_120_20250224_135804', 'trading_data.csv');
const liabilitiesPath = path.join(defaultDownloads, '103901_3_70_20250424_192659', '103901_3_70_20250424_192659_dat.txt');
const segregatedExcelPath = path.join(defaultDownloads, 'SME_Segregated_Data_Cleaned.xlsx');

const outputPath = path.join(projectRoot, 'data', 'processed', 'sme_survival_data.csv');

async function buildSurvivalData() {
  console.log('Ingesting SME raw sources and computing survival trajectories...');
  const underpricedText = fs.readFileSync(underpricedPath, 'utf8');
  const underpricedCodes = new Set(underpricedText.split('\n').map(l => l.trim()).filter(l => l.length > 0));

  const wb = XLSX.readFile(mergedExcelPath);
  const mergedRows = XLSX.utils.sheet_to_json(wb.Sheets['Sheet1']);
  const nameToMerged = new Map();
  for (const row of mergedRows) {
    if (row['Company Name']) {
      const cleanName = row['Company Name'].toLowerCase().replace(/[^a-z0-9]/g, '');
      nameToMerged.set(cleanName, row);
    }
  }

  const wbSeg = XLSX.readFile(segregatedExcelPath);
  const segSheet = wbSeg.Sheets['Sheet1_2'] || wbSeg.Sheets[wbSeg.SheetNames[0]];
  const segRows = XLSX.utils.sheet_to_json(segSheet);
  const nameToSeg = new Map();
  for (const r of segRows) {
    const name = (r['COMPANY NAME'] || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    if (name) nameToSeg.set(name, r);
  }

  const codeToDebt = new Map();
  if (fs.existsSync(liabilitiesPath)) {
    const liabContent = fs.readFileSync(liabilitiesPath, 'utf8');
    for (const line of liabContent.split('\n')) {
      const parts = line.split('|');
      if (parts.length >= 7) {
        const code = parts[0].trim();
        const totLiab = parseFloat(parts[5]);
        const totAssets = parseFloat(parts[6]);
        if (!isNaN(totLiab) && !isNaN(totAssets) && totAssets > 0) {
          codeToDebt.set(code, totLiab / totAssets);
        }
      }
    }
  }

  const coTradingHistory = new Map();
  const coNames = new Map();

  const fileStream = fs.createReadStream(tradingCsvPath);
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let lineCount = 0;
  for await (const line of rl) {
    if (lineCount++ === 0) continue;
    const parts = line.split(',');
    if (parts.length >= 6) {
      const coCode = parts[0].trim();
      const coName = parts[1].trim();
      const stkDate = parts[2].trim();
      const close = parseFloat(parts[3]);
      const qty = parseFloat(parts[4]) || 0;
      const eps = parseFloat(parts[5]) || 0;

      if (!coNames.has(coCode)) coNames.set(coCode, coName);
      if (underpricedCodes.has(coCode)) {
        if (!coTradingHistory.has(coCode)) coTradingHistory.set(coCode, []);
        if (!isNaN(close)) {
          coTradingHistory.get(coCode).push({ date: stkDate, close, qty, eps });
        }
      }
    }
  }

  const compiledRows = [];
  for (const coCode of underpricedCodes) {
    const rawName = coNames.get(coCode) || `Company_${coCode}`;
    const cleanName = rawName.toLowerCase().replace(/[^a-z0-9]/g, '');

    let mergedInfo = nameToMerged.get(cleanName);
    if (!mergedInfo) {
      for (const [k, v] of nameToMerged.entries()) {
        if (cleanName.includes(k) || k.includes(cleanName) || cleanName.substring(0, 8) === k.substring(0, 8)) {
          mergedInfo = v;
          break;
        }
      }
    }

    const segInfo = nameToSeg.get(cleanName);
    const history = coTradingHistory.get(coCode) || [];
    history.sort((a, b) => {
      const parseDate = (s) => {
        const [d, m, y] = s.split('-');
        return new Date(`${y}-${m}-${d}`).getTime();
      };
      return parseDate(a.date) - parseDate(b.date);
    });

    let issuePrice = mergedInfo ? parseFloat(mergedInfo['Issue Price']) : NaN;
    if (isNaN(issuePrice) && segInfo) {
      issuePrice = parseFloat(segInfo['ISSUE PRICE'] || segInfo['PRICE RANGE']);
    }

    let listingClose = mergedInfo ? parseFloat(mergedInfo['Closing Price']) : (history.length > 0 ? history[0].close : NaN);
    let tradedQtyL1 = mergedInfo ? parseFloat(mergedInfo['Traded Quantity']) : (history.length > 0 ? history[0].qty : 0);
    let eps = mergedInfo ? parseFloat(mergedInfo['EPS']) : (history.length > 0 ? history[0].eps : NaN);

    if (isNaN(issuePrice) && history.length > 0) issuePrice = history[0].close * 0.8;

    const listingGainPct = (!isNaN(listingClose) && !isNaN(issuePrice) && issuePrice > 0)
      ? ((listingClose - issuePrice) / issuePrice) * 100
      : 0;

    let event = 0;
    let daysUnderpriced = history.length;
    for (let i = 0; i < history.length; i++) {
      if (history[i].close <= issuePrice) {
        event = 1;
        daysUnderpriced = i + 1;
        break;
      }
    }
    if (daysUnderpriced === 0) daysUnderpriced = 1;

    const debtToAsset = codeToDebt.get(coCode) || 0.45;
    const peRatio = (!isNaN(listingClose) && !isNaN(eps) && eps > 0) ? (listingClose / eps) : 15.0;

    let listingYear = 2023;
    let formattedDate = '';
    const rawDate = mergedInfo ? mergedInfo['Listing Date'] : (history.length > 0 ? history[0].date : '');

    if (typeof rawDate === 'number' || (!isNaN(rawDate) && !String(rawDate).includes('-') && !String(rawDate).includes('/'))) {
      const serial = parseFloat(rawDate);
      if (serial > 20000 && serial < 60000) {
        const utcDays = Math.floor(serial - 25569);
        const dateObj = new Date(utcDays * 86400 * 1000);
        listingYear = dateObj.getUTCFullYear();
        formattedDate = dateObj.toISOString().split('T')[0];
      }
    } else if (rawDate) {
      const dateStr = String(rawDate).trim();
      formattedDate = dateStr;
      const matchYear = dateStr.match(/\b(201\d|202\d)\b/);
      if (matchYear) listingYear = parseInt(matchYear[1]);
      else if (dateStr.includes('-')) {
        const parts = dateStr.split('-');
        if (parts.length === 3) {
          if (parts[2].length === 4) listingYear = parseInt(parts[2]);
          else if (parts[2].length === 2) listingYear = 2000 + parseInt(parts[2]);
          else if (parts[0].length === 4) listingYear = parseInt(parts[0]);
        }
      }
    }

    compiledRows.push({
      co_code: coCode,
      company_name: rawName,
      listing_date: formattedDate,
      listing_year: listingYear,
      issue_price: isNaN(issuePrice) ? 100 : issuePrice,
      listing_close: isNaN(listingClose) ? issuePrice : listingClose,
      listing_gain_pct: listingGainPct.toFixed(2),
      time: daysUnderpriced,
      event: event,
      traded_qty_l1: tradedQtyL1,
      eps: isNaN(eps) ? 5.0 : eps,
      pe_ratio: peRatio.toFixed(2),
      debt_to_asset_ratio: debtToAsset.toFixed(4),
      symbol: segInfo ? (segInfo['Symbol'] || '') : ''
    });
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const header = Object.keys(compiledRows[0]).join(',');
  const csvContent = [header, ...compiledRows.map(r => Object.values(r).map(v => typeof v === 'string' && v.includes(',') ? `"${v}"` : v).join(','))].join('\n');
  fs.writeFileSync(outputPath, csvContent, 'utf8');
  console.log(`Saved ${compiledRows.length} rows to ${outputPath}`);
}

module.exports = { buildSurvivalData };

if (require.main === module) {
  buildSurvivalData().catch(console.error);
}
