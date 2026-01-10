export const legalAdviceMap = {

  kidnapping: {
    categoryLabel: "Kidnapping / Abduction",
    urgency: "EMERGENCY",
    severityLevel: 5,

    legalBasis: {
      law: "BNS",
      sections: "140–151",
      offenceType: "Cognizable & Non-Bailable"
    },

    aiConfidence: "HIGH",

    summary:
      "Unlawful removal or confinement of a person against their will. Extremely time-sensitive.",

    detailedAdvice:
      "Kidnapping cases require immediate police intervention as every minute counts in ensuring victim safety and recovery. Police will launch coordinated search operations, activate surveillance networks, and may involve specialized anti-kidnapping units. If ransom demands are made, do not attempt independent negotiations as this could endanger the victim. Report all communications from kidnappers to police immediately. Time-sensitive evidence like CCTV footage must be preserved within 24-48 hours before it's overwritten. Border alerts and inter-state coordination will be initiated for cases involving movement across jurisdictions.",

    stepsToTake: [
      "Call police emergency number (100)",
      "File complaint at nearest police station",
      "Provide photos and last known location",
      "Preserve ransom or threat evidence"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: true
  },

  sexual_offence: {
    categoryLabel: "Sexual Offences",
    urgency: "HIGH PRIORITY",
    severityLevel: 5,

    legalBasis: {
      law: "BNS",
      sections: "63–70",
      offenceType: "Cognizable & Non-Bailable"
    },

    aiConfidence: "HIGH",

    summary:
      "Crimes violating bodily autonomy and dignity.",

    detailedAdvice:
      "Sexual offence victims have the right to file complaints at ANY police station regardless of jurisdiction under Section 166A CrPC. Female police officers must be made available for statement recording. Medical examination requires victim consent and will be conducted by female doctors at government hospitals. Identity protection is guaranteed by law - no media can disclose victim details. Fast-track courts ensure speedy trials. Free legal aid is available through District Legal Services Authority. Counseling services and support groups can be accessed through Women's Commission. Zero-FIR can be filed if the crime occurred in a different jurisdiction, which will be transferred to the appropriate police station.",

    stepsToTake: [
      "Report immediately",
      "Preserve physical evidence",
      "Request female officer",
      "Seek medical and legal aid"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: true
  },

  assault: {
    categoryLabel: "Assault / Physical Violence",
    urgency: "HIGH",
    severityLevel: 4,

    legalBasis: {
      law: "BNS",
      sections: "115–140",
      offenceType: "Cognizable"
    },

    aiConfidence: "MEDIUM",

    summary:
      "Physical harm or threat of harm to a person.",

    detailedAdvice:
      "Assault charges vary based on injury severity - from simple hurt (BNS 115-117) to grievous hurt (BNS 118-125) to attempt to murder (BNS 109). Immediate medical examination is crucial as medical reports form primary evidence. Document all injuries with photographs from multiple angles. If the assault involved weapons, mention this explicitly in the FIR as it affects charges. For domestic violence cases, approach Women's Commission for protection orders and rehabilitation. Victim compensation schemes may apply for serious injuries. If the accused poses ongoing threat, request police protection or apply for restraining orders through magistrate court.",

    stepsToTake: [
      "Seek medical help",
      "File police complaint",
      "Preserve injury evidence"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: true
  },

  women_child_safety: {
    categoryLabel: "Women & Child Safety",
    urgency: "HIGH PRIORITY",
    severityLevel: 5,

    legalBasis: {
      law: "BNS + POCSO",
      sections: "86 + POCSO Act",
      offenceType: "Cognizable & Non-Bailable"
    },

    aiConfidence: "HIGH",

    summary:
      "Offences involving women or children with enhanced legal protection.",

    detailedAdvice:
      "Cases involving children activate POCSO Act provisions which mandate stringent procedures and fast-track trials. Child-friendly recording of statements is done in special rooms without intimidating environment. For child victims, Child Welfare Committee provides rehabilitation support including counseling, shelter, and education assistance. Women helpline (181) and Child helpline (1098) provide 24/7 assistance. Cases are monitored by District Child Protection Units and State Women's Commissions. Identity protection is absolute - no details can be disclosed publicly. Medical examination and legal aid are provided free of cost. One-Stop Centers (OSC) provide integrated support including police assistance, medical aid, legal counseling, and temporary shelter under one roof.",

    stepsToTake: [
      "Contact police or women/child helpline",
      "Ensure safe environment",
      "Seek legal aid"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: true
  },

  harassment: {
    categoryLabel: "Harassment / Threats / Stalking",
    urgency: "MEDIUM",
    severityLevel: 3,

    legalBasis: {
      law: "BNS",
      sections: "351–353",
      offenceType: "Cognizable"
    },

    aiConfidence: "MEDIUM",

    summary:
      "Repeated threats, intimidation, or unwanted contact.",

    detailedAdvice:
      "Harassment includes verbal abuse, threats, stalking (physical or online), workplace harassment, and eve-teasing. Pattern of behavior is important - maintain detailed log with dates, times, locations, and witnesses for each incident. Save all digital evidence including WhatsApp messages, emails, call recordings, and social media posts as screenshots with timestamps. For workplace harassment, Internal Complaints Committee (ICC) under POSH Act must address complaints within 90 days. For stalking, courts can issue restraining orders prohibiting the accused from approaching within specified distance. Cyber harassment falls under IT Act provisions in addition to BNS sections. Police can issue warning notices to accused which often deters further harassment.",

    stepsToTake: [
      "Document incidents",
      "Preserve messages or calls",
      "File police complaint"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: false
  },

  accident: {
    categoryLabel: "Accident / Hit & Run",
    urgency: "HIGH",
    severityLevel: 4,

    legalBasis: {
      law: "BNS",
      sections: "106, 112, 279",
      offenceType: "Cognizable"
    },

    aiConfidence: "MEDIUM",

    summary:
      "Road accidents involving injury, death, or negligence.",

    detailedAdvice:
      "Traffic accidents require immediate reporting - call 108 for ambulance and 100 for police. For hit and run cases, police will investigate using traffic CCTV footage, toll plaza records, and vehicle tracking systems. Note down vehicle registration number, make, model, and any distinguishing features. Collect witness contact details immediately as they may be difficult to trace later. For serious injuries or death, compensation can be claimed from Motor Accident Claims Tribunal (MACT) within 6 months. Compensation is available from both driver's insurance and government hit-and-run compensation fund. Preserve all medical bills, ambulance receipts, and treatment records for compensation claims. FIR is mandatory for insurance claims even if accident is minor.",

    stepsToTake: [
      "Call emergency services",
      "Assist injured if safe",
      "Report to police"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: true
  },

  cybercrime: {
    categoryLabel: "Cyber Crime",
    urgency: "HIGH",
    severityLevel: 4,

    legalBasis: {
      law: "IT Act + BNS",
      sections: "Various",
      offenceType: "Cognizable"
    },

    aiConfidence: "MEDIUM",

    summary:
      "Online fraud, hacking, identity theft, or harassment.",

    detailedAdvice:
      "Cybercrimes have volatile digital evidence that can be deleted or modified quickly. Do NOT delete any emails, messages, or transaction records even if they seem incriminating. Take full screenshots including timestamps, URLs, and sender details. For financial frauds, immediately inform your bank to freeze accounts or reverse transactions - banks have 4-7 day window for reversals. Report on National Cyber Crime Portal (cybercrime.gov.in) which provides 24/7 online complaint facility. Local Cyber Crime Police Stations have technical expertise for digital forensics including IP address tracking, account tracing, and data recovery. Email headers contain crucial technical information - save complete emails, not just screenshots. For social media impersonation, report to platform immediately and download violation reports. Phishing, UPI frauds, and online scams are prosecuted under IT Act Sections 66, 66C, 66D with imprisonment up to 3 years.",

    stepsToTake: [
      "Preserve digital evidence",
      "Report on cybercrime.gov.in",
      "Inform bank if financial fraud"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: true
  },

  fraud: {
    categoryLabel: "Fraud / Cheating",
    urgency: "MEDIUM",
    severityLevel: 3,

    legalBasis: {
      law: "BNS",
      sections: "318–324",
      offenceType: "Cognizable"
    },

    aiConfidence: "MEDIUM",

    summary:
      "Deception for financial or personal gain.",

    detailedAdvice:
      "Fraud cases involve cheating, forgery, criminal breach of trust, and financial scams. Quick action improves chances of fund recovery and asset attachment. Gather all documentary evidence including agreements, contracts, cheques, receipts, bank statements, and communication records (emails, messages, letters). For check bounce cases, legal notice must be sent within 30 days of dishonor under Negotiable Instruments Act. Large-scale frauds (above Rs 1 crore) are handled by Economic Offences Wing (EOW) which has specialized investigation capabilities. Consumer forums provide alternate remedy for consumer-related frauds with faster resolution. Police can attach properties and freeze bank accounts of accused during investigation. Civil recovery suits can run parallel to criminal cases. Keep certified copies of all documents - never submit originals.",

    stepsToTake: [
      "Collect transaction proof",
      "File police complaint",
      "Inform bank or authority"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: false
  },

  theft: {
    categoryLabel: "Theft / Robbery",
    urgency: "MEDIUM",
    severityLevel: 3,

    legalBasis: {
      law: "BNS",
      sections: "303–309",
      offenceType: "Cognizable"
    },

    aiConfidence: "MEDIUM",

    summary:
      "Dishonest removal of movable property.",

    detailedAdvice:
      "Theft involves dishonest taking without owner's consent while robbery includes violence or threat. File FIR immediately as delay weakens case and affects insurance claims. Prepare detailed list of stolen items with descriptions, serial numbers, IMEI numbers (for phones), and approximate values. Purchase bills and photos of items strengthen evidence. Check surrounding areas for CCTV cameras - shops, ATMs, and residential societies often have coverage. Request police to collect footage within 48 hours before it's overwritten. For vehicle theft, inform RTO to prevent re-registration. Mobile phones can be tracked using IMEI number through Central Equipment Identity Register (CEIR) portal. Inform insurance company within 24-48 hours as per policy terms. Pawn shops and second-hand dealers are often checked by police for stolen goods recovery.",

    stepsToTake: [
      "File FIR",
      "Provide item details",
      "Check CCTV"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: false
  },

  trespass: {
    categoryLabel: "Trespass / Housebreaking",
    urgency: "LOW",
    severityLevel: 2,

    legalBasis: {
      law: "BNS",
      sections: "332–335",
      offenceType: "Cognizable"
    },

    aiConfidence: "LOW",

    summary:
      "Unauthorized entry into property.",

    detailedAdvice:
      "Trespass and housebreaking involve unauthorized entry into property with or without criminal intent. Criminal trespass (BNS 332-333) requires proof of unlawful entry with intent to commit offense, intimidate, or annoy. House-breaking (BNS 334-335) involves forceful entry and carries harsher penalties. Document the trespass with photographs showing entry points, damages, and evidence of intrusion. Collect witness statements from neighbors. For property disputes, civil law remedies run parallel - consult civil lawyer for injunction orders. Keep all property documents (sale deed, title deed, ownership records) ready. Revenue department records from Tehsildar office establish legal ownership. Never resort to violence or forceful eviction as it may result in counter-cases. Security footage is strong evidence if available.",

    stepsToTake: [
      "Document trespass with photos",
      "File police complaint",
      "Gather witness statements",
      "Keep property documents ready"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: false
  },

  defamation: {
    categoryLabel: "Defamation",
    urgency: "LOW",
    severityLevel: 2,

    legalBasis: {
      law: "BNS",
      sections: "356–357",
      offenceType: "Non-Cognizable"
    },

    aiConfidence: "LOW",

    summary:
      "Harm to reputation through false statements.",

    detailedAdvice:
      "Defamation (BNS 356-357) is non-cognizable meaning police cannot arrest without magistrate warrant. Defamation can be both civil (for damages) and criminal (for punishment). False statements must be proven to have damaged reputation in the eyes of reasonable persons. Truth is absolute defense - if statement is true, it's not defamation. Collect all defamatory content with complete context, dates, and publisher details. For print media, keep original copies. For online defamation, take full webpage screenshots with URL and timestamp. Send legal notice demanding apology, retraction, and compensation before filing case - this shows attempt at amicable resolution. Filing is done directly in magistrate court under Section 356 BNS. Witness statements about reputational damage (loss of business, social standing) strengthen case.",

    stepsToTake: [
      "Collect all defamatory material",
      "Take timestamped screenshots",
      "Send legal notice to accused",
      "File complaint in magistrate court",
      "Gather witness statements"
    ],

    canFileComplaintOnline: false,
    requiresImmediatePoliceAction: false
  },

  other: {
    categoryLabel: "Other / Unclassified",
    urgency: "VARIES",
    severityLevel: 1,

    legalBasis: {
      law: "Case Dependent",
      sections: "To be determined",
      offenceType: "Assessment Required"
    },

    aiConfidence: "LOW",

    summary:
      "Complaints not fitting predefined categories.",

    detailedAdvice:
      "Your complaint doesn't clearly fit standard categories but this doesn't diminish its validity. Police are legally obligated to register complaints regardless of classification under Section 154 CrPC. Visit nearest police station with detailed written complaint describing the incident, parties involved, evidence, and relief sought. If police refuse FIR registration, you have right to approach Superintendent of Police (SP) or file complaint under Section 156(3) CrPC in magistrate court which can direct police to investigate. Carry two copies of complaint - one for submission and one for your receipt with acknowledgment. Some cases may require preliminary inquiry to determine cognizable vs non-cognizable nature. Consult with legal aid services or lawyers for proper categorization and appropriate legal remedy.",

    stepsToTake: [
      "Write detailed complaint with all facts",
      "Submit at nearest police station",
      "Request written acknowledgment",
      "If refused, approach SP or magistrate",
      "Consult lawyer for legal categorization"
    ],

    canFileComplaintOnline: true,
    requiresImmediatePoliceAction: false
  }
};
