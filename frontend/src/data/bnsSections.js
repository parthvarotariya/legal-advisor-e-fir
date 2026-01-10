/**
 * Bharatiya Nyaya Sanhita (BNS) 2023 - Section Details
 * 
 * This file contains detailed information about BNS sections
 * for various crime categories used in the e-FIR system.
 */

// Foundation sections (1-10)
export const foundationSections = [
  {
    sectionNumber: "1",
    title: "Short title, extent and commencement",
    description: "This section gives the name Bharatiya Nyaya Sanhita, 2023 and states that it applies to the whole of India. It specifies when the law comes into force.",
    punishment: "N/A"
  },
  {
    sectionNumber: "2",
    title: "Definitions",
    description: "Defines important legal terms used throughout BNS. These definitions ensure crimes and punishments are interpreted uniformly.",
    punishment: "N/A"
  },
  {
    sectionNumber: "3",
    title: "Punishments",
    description: "Lists the types of punishments that courts can impose for offences. Includes imprisonment, fine, death penalty, life imprisonment, and community service.",
    punishment: "N/A"
  },
  {
    sectionNumber: "4",
    title: "Gender",
    description: "Clarifies that words referring to a gender apply to all genders unless specified. Ensures criminal liability is gender-neutral.",
    punishment: "N/A"
  },
  {
    sectionNumber: "5",
    title: "Number",
    description: "Words in singular include plural and vice-versa. Allows offences involving multiple persons or acts to be covered legally.",
    punishment: "N/A"
  },
  {
    sectionNumber: "6",
    title: "Person",
    description: "Defines 'person' to include individuals, companies, associations, and bodies of persons. This allows corporations and groups to be held criminally liable.",
    punishment: "N/A"
  },
  {
    sectionNumber: "7",
    title: "Public Servant",
    description: "Explains who is considered a public servant under criminal law. Used for offences involving corruption, misuse of power, or duty violations.",
    punishment: "N/A"
  },
  {
    sectionNumber: "8",
    title: "Voluntarily",
    description: "An act is voluntary if it is done knowingly and intentionally. Criminal liability generally requires conscious action.",
    punishment: "N/A"
  },
  {
    sectionNumber: "9",
    title: "Dishonestly",
    description: "An act is dishonest if done with intent to cause wrongful gain or loss. This concept is central to theft, cheating, and fraud offences.",
    punishment: "N/A"
  },
  {
    sectionNumber: "10",
    title: "Fraudulently",
    description: "An act is fraudulent if done with intent to deceive. Used in offences involving cheating, forgery, and financial crimes.",
    punishment: "N/A"
  }
];

export const bnsSections = {
  
  // Kidnapping / Abduction / Missing Person
  kidnapping: {
    sections: [
      {
        sectionNumber: "140",
        title: "Kidnapping",
        description: "Whoever conveys any person beyond the limits of India without the consent of that person, or of some person legally authorized to consent on behalf of that person, is said to kidnap that person from India.",
        punishment: "Imprisonment up to 7 years and fine"
      },
      {
        sectionNumber: "141",
        title: "Kidnapping from lawful guardianship",
        description: "Whoever takes or entices any minor under 18 years of age if a male, or under 18 years of age if a female, or any person of unsound mind, out of the keeping of the lawful guardian of such minor or person of unsound mind.",
        punishment: "Imprisonment up to 7 years and fine"
      },
      // Add more sections 142-151 here
    ]
  },

  // Sexual Offences
  sexual_offence: {
    sections: [
      {
        sectionNumber: "63",
        title: "Rape",
        description: "A man is said to commit rape if he has sexual intercourse with a woman under circumstances falling under any of the specified categories without her consent or with consent obtained by force, threat, fraud, or intoxication.",
        punishment: "Rigorous imprisonment not less than 10 years, extendable to life imprisonment and fine"
      },
      {
        sectionNumber: "64",
        title: "Rape causing death or persistent vegetative state",
        description: "Whoever commits rape and causes death or persistent vegetative state to the victim.",
        punishment: "Rigorous imprisonment not less than 20 years to life imprisonment or death"
      },
      // Add more sections 65-70 here
    ]
  },

  // Assault / Hurt / Violence
  assault: {
    sections: [
      {
        sectionNumber: "115",
        title: "Voluntarily causing hurt",
        description: "Whoever does any act with the intention of thereby causing hurt to any person, or with the knowledge that he is likely thereby to cause hurt to any person, and does thereby cause hurt to any person.",
        punishment: "Imprisonment up to 1 year or fine up to Rs. 10,000 or both"
      },
      {
        sectionNumber: "118",
        title: "Voluntarily causing grievous hurt",
        description: "Whoever voluntarily causes hurt, if the hurt which he intends to cause or knows himself to be likely to cause is grievous hurt.",
        punishment: "Imprisonment up to 7 years and fine"
      },
      // Add more sections 119-140 here
    ]
  },

  // Women & Child Safety
  women_child_safety: {
    sections: [
      {
        sectionNumber: "86",
        title: "Offence committed on woman",
        description: "Special provisions for offences committed specifically against women.",
        punishment: "As per specific offence provisions"
      }
      // Add POCSO Act sections if needed
    ]
  },

  // Harassment / Threats / Stalking
  harassment: {
    sections: [
      {
        sectionNumber: "351",
        title: "Criminal intimidation",
        description: "Whoever threatens another with any injury to his person, reputation or property, or to the person or reputation of any one in whom that person is interested, with intent to cause alarm to that person.",
        punishment: "Imprisonment up to 2 years or fine or both"
      },
      {
        sectionNumber: "352",
        title: "Intentional insult with intent to provoke breach of peace",
        description: "Whoever intentionally insults and thereby gives provocation to any person, intending or knowing it to be likely that such provocation will cause breach of peace.",
        punishment: "Imprisonment up to 2 years or fine or both"
      },
      {
        sectionNumber: "353",
        title: "Stalking",
        description: "Following, contacting, or attempting to contact a person to foster personal interaction repeatedly despite clear indication of disinterest, or monitoring use of internet or electronic communication.",
        punishment: "Imprisonment up to 3 years and fine; subsequent offence up to 5 years and fine"
      }
    ]
  },

  // Accident / Hit & Run
  accident: {
    sections: [
      {
        sectionNumber: "106",
        title: "Causing death by negligence",
        description: "Whoever causes death of any person by doing any rash or negligent act not amounting to culpable homicide.",
        punishment: "Imprisonment up to 5 years and fine"
      },
      {
        sectionNumber: "112",
        title: "Punishment for causing hurt by act endangering life or personal safety of others",
        description: "Whoever does any act so rashly or negligently as to endanger human life or the personal safety of others.",
        punishment: "Imprisonment up to 6 months or fine up to Rs. 1,000 or both"
      },
      {
        sectionNumber: "279",
        title: "Rash driving or riding on a public way",
        description: "Whoever drives any vehicle, or rides, on any public way in a manner so rash or negligent as to endanger human life.",
        punishment: "Imprisonment up to 6 months or fine up to Rs. 1,000 or both"
      }
    ]
  },

  // Cybercrime (IT Act + BNS mapping)
  cybercrime: {
    sections: [
      {
        sectionNumber: "IT Act 66",
        title: "Computer related offences",
        description: "If any person, dishonestly or fraudulently, does any act referred to in section 43, he shall be punishable with imprisonment up to three years or with fine which may extend to five lakh rupees or with both.",
        punishment: "Imprisonment up to 3 years or fine up to Rs. 5 lakhs or both"
      },
      {
        sectionNumber: "IT Act 66C",
        title: "Identity theft",
        description: "Whoever, fraudulently or dishonestly make use of the electronic signature, password or any other unique identification feature of any other person.",
        punishment: "Imprisonment up to 3 years and fine up to Rs. 1 lakh"
      },
      {
        sectionNumber: "IT Act 66D",
        title: "Cheating by personation using computer resource",
        description: "Whoever, by means of any communication device or computer resource cheats by personation.",
        punishment: "Imprisonment up to 3 years and fine up to Rs. 1 lakh"
      }
    ]
  },

  // Fraud / Cheating / Financial Crimes
  fraud: {
    sections: [
      {
        sectionNumber: "318",
        title: "Cheating",
        description: "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person, or to consent that any person shall retain any property.",
        punishment: "Imprisonment up to 3 years or fine or both"
      },
      {
        sectionNumber: "319",
        title: "Cheating by personation",
        description: "A person is said to cheat by personation if he cheats by pretending to be some other person, or by knowingly substituting one person for another.",
        punishment: "Imprisonment up to 5 years or fine or both"
      },
      {
        sectionNumber: "320",
        title: "Cheating and dishonestly inducing delivery of property",
        description: "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person.",
        punishment: "Imprisonment up to 7 years and fine"
      }
      // Add more sections 321-324
    ]
  },

  // Theft & Robbery
  theft: {
    sections: [
      {
        sectionNumber: "303",
        title: "Theft",
        description: "Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property in order to such taking.",
        punishment: "Imprisonment up to 3 years or fine or both"
      },
      {
        sectionNumber: "304",
        title: "Theft in dwelling house",
        description: "Whoever commits theft in any building, tent or vessel used as a human dwelling.",
        punishment: "Imprisonment up to 7 years and fine"
      },
      {
        sectionNumber: "309",
        title: "Robbery",
        description: "Theft is robbery if, in order to the committing of the theft, or in committing the theft, or in carrying away or attempting to carry away property obtained by the theft, the offender, for that end, voluntarily causes or attempts to cause to any person death or hurt or wrongful restraint, or fear of instant death or of instant hurt, or of instant wrongful restraint.",
        punishment: "Rigorous imprisonment up to 10 years and fine"
      }
      // Add more sections 305-308
    ]
  },

  // Trespass / Housebreaking / Property Disputes
  trespass: {
    sections: [
      {
        sectionNumber: "332",
        title: "House-trespass",
        description: "Whoever commits criminal trespass by entering into or remaining in any building, tent or vessel used as a human dwelling.",
        punishment: "Imprisonment up to 1 year or fine up to Rs. 5,000 or both"
      },
      {
        sectionNumber: "333",
        title: "House-trespass in order to commit offence",
        description: "Whoever commits house-trespass in order to the committing of any offence punishable with death.",
        punishment: "Imprisonment up to 10 years and fine"
      },
      {
        sectionNumber: "334",
        title: "House-breaking",
        description: "A person is said to commit house-breaking who commits house-trespass if he effects his entrance into the house or any part of it in any of the six ways specified.",
        punishment: "Imprisonment up to 2 years and fine"
      },
      {
        sectionNumber: "335",
        title: "House-breaking by night",
        description: "Whoever commits house-breaking after sunset and before sunrise.",
        punishment: "Imprisonment up to 3 years and fine"
      }
    ]
  },

  // Defamation / Public Order Offences
  defamation: {
    sections: [
      {
        sectionNumber: "356",
        title: "Defamation",
        description: "Whoever, by words either spoken or intended to be read, or by signs or by visible representations, makes or publishes any imputation concerning any person intending to harm, or knowing or having reason to believe that such imputation will harm, the reputation of such person.",
        punishment: "Imprisonment up to 2 years or fine or both"
      },
      {
        sectionNumber: "357",
        title: "Printing or engraving matter known to be defamatory",
        description: "Whoever prints or engraves any matter, knowing or having good reason to believe that such matter is defamatory of any person.",
        punishment: "Imprisonment up to 2 years or fine or both"
      }
      // Add more sections 147-150 for public order offences if needed
    ]
  },

  // Other / General Provisions
  other: {
    sections: [
      {
        sectionNumber: "Various",
        title: "Case-dependent provisions",
        description: "Applicable sections will be determined based on the specific nature of the complaint after preliminary inquiry.",
        punishment: "As per applicable provisions"
      }
    ]
  }

};

/**
 * Get BNS sections for a specific crime category
 * @param {string} category - Crime category key
 * @returns {Array} Array of section objects
 */
export const getBnsSections = (category) => {
  return bnsSections[category] || bnsSections.other;
};

/**
 * Get all BNS sections as flat array
 * @returns {Array} All sections from all categories
 */
export const getAllBnsSections = () => {
  const allSections = [];
  Object.keys(bnsSections).forEach(category => {
    if (bnsSections[category].sections) {
      allSections.push(...bnsSections[category].sections);
    }
  });
  return allSections;
};

/**
 * Search BNS sections by keyword
 * @param {string} keyword - Search term
 * @returns {Array} Matching sections
 */
export const searchBnsSections = (keyword) => {
  const allSections = getAllBnsSections();
  const lowerKeyword = keyword.toLowerCase();
  
  return allSections.filter(section => 
    section.title.toLowerCase().includes(lowerKeyword) ||
    section.description.toLowerCase().includes(lowerKeyword) ||
    section.sectionNumber.toLowerCase().includes(lowerKeyword)
  );
};
