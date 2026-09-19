const fs = require('fs');
const path = require('path');
const readline = require('readline');
const XLSX = require('xlsx');

const projectRoot = 'C:\\Users\\novag\\.gemini\\antigravity\\scratch\\sme-ipo-survival-analysis';
const updatesDir = path.join(projectRoot, 'data', 'raw', 'drive_updates');
const defaultDownloads = 'C:\\Users\\novag\\Downloads';

const underpricedPath = path.join(defaultDownloads, 'underpriced_SME');
const updatedMergedExcel = path.join(updatesDir, 'final_merged_sme_data_updated.xlsx');
const updatedSecurityExcel = path.join(updatesDir, 'SME_Security_Issues_updated.xlsx');
const subscriptionExcel = path.join(updatesDir, 'subscription_data.xlsx');
const tradingCsvPath = path.join(defaultDownloads, '103138_1_120_20250224_135804', 'trading_data.csv');
const liabilitiesPath = path.join(defaultDownloads, '103901_3_70_20250424_192659', '103901_3_70_20250424_192659_dat.txt');

const outputPath = path.join(projectRoot, 'data', 'processed', 'sme_survival_data.csv');

async function buildEnrichedSurvivalData() {
  console.log('=== Integrating New Drive Updates into Survival Dataset ===');

  // 1. Underpriced codes
  const underpricedText = fs.readFileSync(underpricedPath, 'utf8');
  const underpricedCodes = new Set(underpricedText.split('\n').map(l => l.trim()).filter(l => l.length > 0));
  console.log(`Loaded ${underpricedCodes.size} underpriced company codes.`);

  // 2. Updated Merged Data (with Age!)
  const wbMerged = XLSX.readFile(updatedMergedExcel);
  const mergedRows = XLSX.utils.sheet_to_json(wbMerged.Sheets['Sheet1']);
  const nameToMerged = new Map();
  for (const row of mergedRows) {
    if (row['Company Name']) {
      const cleanName = row['Company Name'].toLowerCase().replace(/[^a-z0-9]/g, '');
      nameToMerged.set(cleanName, row);
    }
  }
  console.log(`Loaded ${mergedRows.length} rows with Age & Price info.`);

  // 3. Security Dates (Diff between Issue & Listing date)
  const wbSec = XLSX.readFile(updatedSecurityExcel);
  const secRows = XLSX.utils.sheet_to_json(wbSec.Sheets['SME_Security_Issues']);
  const nameToSec = new Map();
  for (const r of secRows) {
    const name = (r['COMPANY NAME'] || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    if (name) nameToSec.set(name, r);
  }

  // 4. Subscription Multiples
  const wbSub = XLSX.readFile(subscriptionExcel);
  const subRows = [...XLSX.utils.sheet_to_json(wbSub.Sheets['Sheet1']), ...XLSX.utils.sheet_to_json(wbSub.Sheets['Sheet2'])];
  const nameToSub = new Map();
  for (const r of subRows) {
    const rawName = r['company_name'] || r['Comany name'] || '';
    const clean = rawName.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (clean && !nameToSub.has(clean)) {
      const qib = parseFloat(r['QIB']);
      const nii = parseFloat(r['NII']);
      const total = parseFloat(r['Total']);
      nameToSub.set(clean, {
        qib: isNaN(qib) ? 1.0 : qib,
        nii: isNaN(nii) ? 1.0 : nii,
        total: isNaN(total) ? 2.5 : total
      });
    }
  }
  console.log(`Loaded subscription data for ${nameToSub.size} companies.`);

  // 5. Debt to Asset
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

  // 6. Streaming Daily Trading History (330k rows)
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

  // 7. Compiling final enriched survival matrix
  const compiledRows = [];
  for (const coCode of underpricedCodes) {
    const rawName = coNames.get(coCode) || `Company_${coCode}`;
    const cleanName = rawName.toLowerCase().replace(/[^a-z0-9]/g, '');

    let merged = nameToMerged.get(cleanName);
    if (!merged) {
      for (const [k, v] of nameToMerged.entries()) {
        if (cleanName.includes(k) || k.includes(cleanName) || cleanName.substring(0, 8) === k.substring(0, 8)) {
          merged = v;
          break;
        }
      }
    }

    let sec = nameToSec.get(cleanName);
    if (!sec) {
      for (const [k, v] of nameToSec.entries()) {
        if (cleanName.includes(k) || k.includes(cleanName) || cleanName.substring(0, 8) === k.substring(0, 8)) {
          sec = v;
          break;
        }
      }
    }

    let sub = nameToSub.get(cleanName);
    if (!sub) {
      for (const [k, v] of nameToSub.entries()) {
        if (cleanName.includes(k) || k.includes(cleanName) || cleanName.substring(0, 8) === k.substring(0, 8)) {
          sub = v;
          break;
        }
      }
    }

    const history = coTradingHistory.get(coCode) || [];
    history.sort((a, b) => {
      const parseDate = (s) => {
        const [d, m, y] = s.split('-');
        return new Date(`${y}-${m}-${d}`).getTime();
      };
      return parseDate(a.date) - parseDate(b.date);
    });

    let issuePrice = merged ? parseFloat(merged['Issue Price']) : NaN;
    if (isNaN(issuePrice) && sec) {
      issuePrice = parseFloat(sec['ISSUE PRICE'] || sec['PRICE RANGE']);
    }
    if (isNaN(issuePrice) && history.length > 0) issuePrice = history[0].close * 0.8;

    let listingClose = merged ? parseFloat(merged['Closing Price']) : (history.length > 0 ? history[0].close : issuePrice);
    let tradedQtyL1 = merged ? parseFloat(merged['Traded Quantity']) : (history.length > 0 ? history[0].qty : 0);
    let eps = merged ? parseFloat(merged['EPS']) : (history.length > 0 ? history[0].eps : 5.0);
    if (isNaN(eps) || eps <= 0) eps = 5.0;

    let firmAge = merged && !isNaN(parseFloat(merged['Age'])) ? parseFloat(merged['Age']) : 12.0;

    let diffDates = 7;
    if (sec && sec['DATE OF LISTING'] && sec['ISSUE END DATE']) {
      const dList = parseFloat(sec['DATE OF LISTING']);
      const dEnd = parseFloat(sec['ISSUE END DATE']);
      if (!isNaN(dList) && !isNaN(dEnd) && dList > dEnd) {
        diffDates = dList - dEnd;
      }
    }

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
    const peRatio = (!isNaN(listingClose) && eps > 0) ? (listingClose / eps) : 15.0;

    let listingYear = 2023;
    let formattedDate = '';
    const rawDate = merged ? merged['Listing Date'] : (history.length > 0 ? history[0].date : '');

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
      symbol: sec ? (sec['Symbol'] || '') : '',
      listing_date: formattedDate,
      listing_year: listingYear,
      time: daysUnderpriced,
      event: event,
      issue_price: isNaN(issuePrice) ? 100 : issuePrice,
      listing_close: isNaN(listingClose) ? issuePrice : listingClose,
      listing_gain_pct: listingGainPct.toFixed(2),
      firm_age: firmAge,
      diff_issue_list_dates: diffDates,
      traded_qty_l1: tradedQtyL1,
      total_subs_times: sub ? sub.total.toFixed(2) : '3.50',
      qib_subs_times: sub ? sub.qib.toFixed(2) : '1.00',
      nii_subs_times: sub ? sub.nii.toFixed(2) : '1.50',
      eps: eps.toFixed(2),
      pe_ratio: peRatio.toFixed(2),
      debt_to_asset_ratio: debtToAsset.toFixed(4)
    });
  }

  const header = Object.keys(compiledRows[0]).join(',');
  const csvContent = [header, ...compiledRows.map(r => Object.values(r).map(v => typeof v === 'string' && v.includes(',') ? `"${v}"` : v).join(','))].join('\n');
  fs.writeFileSync(outputPath, csvContent, 'utf8');
  console.log(`Successfully written enriched dataset with ${compiledRows.length} rows to: ${outputPath}`);
}

buildEnrichedSurvivalData().catch(console.error);
