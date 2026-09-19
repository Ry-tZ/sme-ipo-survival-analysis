const fs = require('fs');
const path = require('path');

const csvPath = 'C:\\Users\\novag\\.gemini\\antigravity\\scratch\\sme-ipo-survival-analysis\\data\\processed\\sme_survival_data.csv';

function testDataIntegrity() {
  console.log('Running Data Integrity Tests on compiled survival dataset...');
  const content = fs.readFileSync(csvPath, 'utf8');
  const lines = content.split('\n').filter(l => l.trim().length > 0);
  
  const header = lines[0].split(',');
  console.log(`[PASS] Headers present (${header.length} columns): ${header.join(', ')}`);
  
  const timeIdx = header.indexOf('time');
  const eventIdx = header.indexOf('event');
  
  if (timeIdx === -1 || eventIdx === -1) {
    throw new Error('Missing time or event column');
  }
  
  const rows = lines.slice(1);
  if (rows.length !== 436) {
    throw new Error(`Expected 436 rows, found ${rows.length}`);
  }
  console.log(`[PASS] Row count matches target cohort: 436 underpriced companies.`);
  
  let validTimes = 0;
  let validEvents = 0;
  let event1Count = 0;
  let event0Count = 0;
  
  for (let i = 0; i < rows.length; i++) {
    const parts = rows[i].split(',');
    const time = parseFloat(parts[timeIdx]);
    const event = parseInt(parts[eventIdx]);
    
    if (!isNaN(time) && time >= 1) validTimes++;
    if (event === 0 || event === 1) {
      validEvents++;
      if (event === 1) event1Count++;
      else event0Count++;
    }
  }
  
  if (validTimes !== 436) throw new Error(`Invalid time values: ${validTimes}/436`);
  console.log(`[PASS] All 436 survival times are valid positive durations (min 1 day).`);
  
  if (validEvents !== 436) throw new Error(`Invalid event flags: ${validEvents}/436`);
  console.log(`[PASS] All 436 event indicators are strictly binary: ${event1Count} uncensored (${((event1Count/436)*100).toFixed(1)}%), ${event0Count} right-censored (${((event0Count/436)*100).toFixed(1)}%).`);
  
  console.log('\n>>> ALL DATA INTEGRITY TESTS PASSED SUCCESSFULLY! <<<');
}

testDataIntegrity();
