import { jsPDF } from 'jspdf';

/**
 * Sanitize text for jsPDF — replace Unicode chars not in Latin-1
 * jsPDF built-in fonts (Helvetica/Times/Courier) only support Latin-1.
 * Characters like ₹, —, ", ", ', ' etc. cause garbled/spaced-out rendering.
 */
const sanitize = (text) => {
  if (!text) return '';
  return String(text)
    .replace(/₹/g, 'Rs.')
    .replace(/—/g, '-')
    .replace(/–/g, '-')
    .replace(/"/g, '"')
    .replace(/"/g, '"')
    .replace(/'/g, "'")
    .replace(/'/g, "'")
    .replace(/…/g, '...')
    .replace(/•/g, '*')
    .replace(/✅/g, '[Y]')
    .replace(/❌/g, '[N]')
    .replace(/[^\x00-\xFF]/g, ''); // strip anything else outside Latin-1
};

/**
 * Generate an official FIR PDF matching the Indian First Information Report format
 * Under Section 173 B.N.S.S (Bharatiya Nagarik Suraksha Sanhita)
 */
export const generateFirPdf = (fir) => {
  const doc = new jsPDF('p', 'mm', 'a4');
  const pageWidth = 210;
  const margin = 15;
  const contentWidth = pageWidth - 2 * margin;
  let y = 12;

  const addPage = () => {
    doc.addPage();
    y = 15;
  };

  const checkPage = (needed = 20) => {
    if (y + needed > 280) addPage();
  };

  /* -- Helpers -- */
  const bold = (size = 10) => doc.setFont('helvetica', 'bold').setFontSize(size);
  const normal = (size = 10) => doc.setFont('helvetica', 'normal').setFontSize(size);
  const italic = (size = 9) => doc.setFont('helvetica', 'italic').setFontSize(size);

  const drawLine = (x1, y1, x2, y2) => {
    doc.setDrawColor(0);
    doc.setLineWidth(0.3);
    doc.line(x1, y1, x2, y2);
  };

  const drawBox = (x, yy, w, h) => {
    doc.setDrawColor(0);
    doc.setLineWidth(0.3);
    doc.rect(x, yy, w, h);
  };

  const labelValue = (lbl, val, x, yy, lblWidth = 45) => {
    bold(9);
    doc.text(sanitize(lbl), x, yy);
    normal(9);
    doc.text(sanitize(val) || '-', x + lblWidth, yy);
  };

  const formatDate = (d) => {
    if (!d) return '-';
    try {
      const date = new Date(d);
      return date.toLocaleDateString('en-IN', { day: '2-digit', month: '2-digit', year: 'numeric' });
    } catch { return String(d); }
  };

  const formatTime = (t) => {
    if (!t) return '-';
    return String(t);
  };

  const wrapText = (text, x, yy, maxWidth, lineHeight = 4.5) => {
    if (!text) return yy;
    const clean = sanitize(text);
    const lines = doc.splitTextToSize(clean, maxWidth);
    for (const line of lines) {
      checkPage(lineHeight + 2);
      doc.text(line, x, yy);
      yy += lineHeight;
    }
    return yy;
  };

  /* ══════════════════════════════════════════════════
     PAGE 1: HEADER + BASIC INFO
     ══════════════════════════════════════════════════ */

  // Outer border
  drawBox(margin - 2, 8, contentWidth + 4, 280);

  // Title
  bold(14);
  doc.text('FIRST INFORMATION REPORT', pageWidth / 2, y, { align: 'center' });
  y += 6;
  bold(10);
  doc.text('(Under Section 173 B.N.S.S)', pageWidth / 2, y, { align: 'center' });
  y += 5;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 1. District / PS / Year / FIR No / Date ── */
  bold(9);
  doc.text('1', margin, y);

  const col1 = margin + 8;
  const col2 = margin + 55;
  const col3 = margin + 105;
  const col4 = margin + 145;

  labelValue('District:', fir.district, col1, y, 18);
  labelValue('Police Station:', fir.policeStationName, col2, y, 30);
  y += 5;
  labelValue('FIR Number:', fir.firNumber, col1, y, 25);
  labelValue('Date:', formatDate(fir.registeredAt), col3, y, 14);
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 2. Act / Sections ── */
  bold(9);
  doc.text('2', margin, y);
  labelValue('(i) Act:', 'BNS (Bharatiya Nyaya Sanhita)', col1, y, 14);
  labelValue('Sections:', fir.ipcSections || '—', col3, y, 20);
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 3. Occurrence of Offence ── */
  bold(9);
  doc.text('3', margin, y);
  bold(9);
  doc.text('(a) Occurrence of Offence:', col1, y);
  y += 6;

  labelValue('Date:', formatDate(fir.incidentDate), col1 + 5, y, 12);
  labelValue('Time:', formatTime(fir.incidentTime), col3, y, 12);
  y += 6;

  bold(9);
  doc.text('(b) Information received at PS:', col1, y);
  y += 5;
  labelValue('Date:', formatDate(fir.registeredAt), col1 + 5, y, 12);
  labelValue('Time:', formatTime(fir.registeredAt ? new Date(fir.registeredAt).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : ''), col3, y, 12);
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 4. Type of Information ── */
  bold(9);
  doc.text('4', margin, y);
  labelValue('Type of Information:', fir.isEfir ? 'e-FIR (Electronic)' : 'Written / Oral', col1, y, 38);
  if (fir.isZeroFir) {
    y += 5;
    labelValue('Zero FIR:', 'Yes — Destination PS: ' + (fir.destinationPoliceStation || '—'), col1, y, 20);
  }
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 5. Place of Occurrence ── */
  bold(9);
  doc.text('5', margin, y);
  bold(9);
  doc.text('Place of Occurrence:', col1, y);
  y += 6;

  labelValue('(a) Address:', fir.incidentLocation || '—', col1 + 5, y, 22);
  y += 5;
  labelValue('(b) District:', fir.district || '—', col1 + 5, y, 22);
  if (fir.policeStationName) {
    y += 5;
    labelValue('Police Station:', fir.policeStationName, col1 + 5, y, 30);
  }
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 6. Complainant / Informant ── */
  bold(9);
  doc.text('6', margin, y);
  bold(9);
  doc.text('Complainant / Informant:', col1, y);
  y += 6;

  labelValue('(a) Name:', fir.informantName || '—', col1 + 5, y, 20);
  y += 5;
  labelValue('(b) Father / Guardian:', fir.informantGuardianName || '—', col1 + 5, y, 38);
  y += 5;
  labelValue('(c) Contact:', fir.informantContact || '—', col1 + 5, y, 22);
  labelValue('Email:', fir.informantEmail || '—', col3, y, 14);
  y += 5;
  labelValue('(d) Address:', '', col1 + 5, y, 22);
  y += 4;
  normal(9);
  y = wrapText(fir.informantAddress || '—', col1 + 27, y, contentWidth - 40);
  y += 3;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 7. Details of Accused ── */
  checkPage(25);
  bold(9);
  doc.text('7', margin, y);
  bold(9);
  doc.text('Details of known/suspected/unknown accused:', col1, y);
  y += 6;
  normal(9);
  if (fir.accusedDetails) {
    y = wrapText(fir.accusedDetails, col1 + 5, y, contentWidth - 20);
  } else {
    doc.text('Not available / Unknown', col1 + 5, y);
    y += 5;
  }
  y += 3;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 8. Reasons for delay ── */
  checkPage(15);
  bold(9);
  doc.text('8', margin, y);
  labelValue('Reasons for delay in reporting:', 'N/A', col1, y, 55);
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 9. Stolen Property ── */
  checkPage(20);
  bold(9);
  doc.text('9', margin, y);
  bold(9);
  doc.text('Particulars of properties stolen (if any):', col1, y);
  y += 5;
  normal(9);
  if (fir.stolenPropertyDetails) {
    y = wrapText(fir.stolenPropertyDetails, col1 + 5, y, contentWidth - 20);
  } else {
    doc.text('N/A', col1 + 5, y);
    y += 5;
  }
  y += 3;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 10. Total value ── */
  checkPage(12);
  bold(9);
  doc.text('10', margin, y);
  labelValue('Total value of property stolen:', 'N/A', col1, y, 55);
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 11. Inquest Report ── */
  checkPage(12);
  bold(9);
  doc.text('11', margin, y);
  labelValue('Inquest Report / U.D. case No.:', 'N/A', col1, y, 55);
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 12. First Information Contents ── */
  checkPage(30);
  bold(9);
  doc.text('12', margin, y);
  bold(9);
  doc.text('First Information Contents:', col1, y);
  y += 2;

  // Crime Category
  y += 5;
  bold(8);
  doc.text('Crime Category: ', col1 + 5, y);
  normal(8);
  doc.text(sanitize(fir.crimeCategory) || '-', col1 + 35, y);
  y += 5;

  // Complaint / Incident Description
  bold(8);
  doc.text('Complaint:', col1 + 5, y);
  y += 4;
  normal(8);
  const descText = fir.incidentDescription || fir.complaintDescription || 'No description provided.';
  y = wrapText(descText, col1 + 5, y, contentWidth - 20, 4);

  // Witness details
  if (fir.witnessDetails) {
    y += 4;
    checkPage(15);
    bold(8);
    doc.text('Witness Details:', col1 + 5, y);
    y += 4;
    normal(8);
    y = wrapText(fir.witnessDetails, col1 + 5, y, contentWidth - 20, 4);
  }

  y += 5;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 13. Action Taken ── */
  checkPage(35);
  bold(9);
  doc.text('13', margin, y);
  bold(9);
  doc.text('Action Taken:', col1, y);
  y += 6;
  normal(9);
  doc.text('(1) Registered the case and took up the investigation.', col1 + 5, y);
  y += 6;

  if (fir.investigatingOfficerName) {
    doc.text('(2) Directed (Name of I.O.) to take up the investigation:', col1 + 5, y);
    y += 5;
    bold(9);
    const ioName = sanitize(fir.investigatingOfficerName) || '-';
    doc.text(ioName, col1 + 10, y);
    normal(9);
    if (fir.investigatingOfficerRank) {
      doc.text('   Rank: ' + sanitize(fir.investigatingOfficerRank), col1 + 10 + doc.getTextWidth(ioName) + 2, y);
    }
    if (fir.investigatingOfficerBadgeNumber) {
      y += 5;
      doc.text('Badge No.: ' + sanitize(fir.investigatingOfficerBadgeNumber), col1 + 10, y);
    }
    y += 5;
  }

  y += 3;
  normal(8);
  doc.text('F.I.R. read over to the complainant/informant, admitted to be correctly recorded', col1 + 5, y);
  y += 4;
  doc.text('and a copy given to the complainant/informant, free of cost.', col1 + 5, y);
  y += 7;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 14. Signatures ── */
  checkPage(40);
  bold(9);
  doc.text('14', margin, y);
  y += 2;

  // Right side: Officer signature
  bold(9);
  doc.text('Signature of Officer in Charge,', col3, y);
  y += 5;
  doc.text('Police Station', col3, y);
  y += 8;
  drawLine(col3, y, pageWidth - margin, y);
  y += 5;
  normal(9);
  doc.text('Name: ' + sanitize(fir.firWrittenBy || '-'), col3, y);
  y += 5;
  if (fir.policeStationName) {
    doc.text('PS: ' + sanitize(fir.policeStationName), col3, y);
    y += 5;
  }

  // Left side: Complainant
  const sigY = y - 18;
  bold(9);
  doc.text('Signature / Thumb Impression', col1, sigY);
  y = sigY + 4;
  doc.text('of the Complainant / Informant', col1, y);
  y += 10;
  drawLine(col1, y, col1 + 50, y);
  y += 5;
  normal(9);
  doc.text('Name: ' + sanitize(fir.informantName || '-'), col1, y);

  y += 10;
  drawLine(margin, y, pageWidth - margin, y);
  y += 6;

  /* ── 15. Dispatch ── */
  checkPage(15);
  bold(9);
  doc.text('15', margin, y);
  labelValue('Date and time of dispatch to the court:', formatDate(fir.registeredAt) + '  ' + (fir.registeredAt ? new Date(fir.registeredAt).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : ''), col1, y, 68);
  y += 8;

  /* ── BNSS 2023 Compliance Note ── */
  if (fir.isEfir || fir.isZeroFir || fir.isVictimWoman || fir.isDisabledVictim) {
    checkPage(30);
    drawLine(margin, y, pageWidth - margin, y);
    y += 6;
    bold(9);
    doc.text('BNSS 2023 Compliance Details:', col1, y);
    y += 5;
    normal(8);
    if (fir.isEfir) { doc.text('* e-FIR: Yes (Filed electronically)', col1 + 5, y); y += 4; }
    if (fir.isZeroFir) { doc.text('* Zero FIR: Yes - Destination PS: ' + sanitize(fir.destinationPoliceStation || '-'), col1 + 5, y); y += 4; }
    if (fir.isVictimWoman) { doc.text('* Woman Victim: Yes', col1 + 5, y); y += 4; }
    if (fir.recordedByWomanOfficer) { doc.text('* Recorded by Woman Officer: Yes', col1 + 5, y); y += 4; }
    if (fir.isDisabledVictim) { doc.text('* Disabled Victim: Yes', col1 + 5, y); y += 4; }
    if (fir.interpreterOrEducatorName) { doc.text('* Interpreter/Educator: ' + sanitize(fir.interpreterOrEducatorName), col1 + 5, y); y += 4; }
    if (fir.isMagistrateStatementRecorded) { doc.text('* Magistrate Statement: Recorded', col1 + 5, y); y += 4; }
  }

  /* ── Footer ── */
  const totalPages = doc.internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    normal(7);
    doc.setTextColor(128);
    doc.text(`Page ${i} of ${totalPages}`, pageWidth / 2, 292, { align: 'center' });
    doc.text('Generated from Legal Advisor e-FIR System', pageWidth / 2, 296, { align: 'center' });
    doc.setTextColor(0);
  }

  /* ── Save ── */
  const filename = `FIR_${(fir.firNumber || 'UNKNOWN').replace(/[^a-zA-Z0-9-]/g, '_')}.pdf`;
  doc.save(filename);
};

export default generateFirPdf;
