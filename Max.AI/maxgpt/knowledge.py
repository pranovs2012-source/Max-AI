"""World-knowledge facts for MaxGPT, turned into training conversations.

Edit the tables below (keep them accurate!) and regenerate the chat file:

    python -m maxgpt.knowledge          # writes data/chat/20_world_facts.txt

Each fact becomes several conversations with different wordings, so the model
learns to answer however the question is asked.
"""
import os

# country: (capital, continent, currency, main language(s))
COUNTRIES = {
    "France": ("Paris", "Europe", "the euro", "French"),
    "Germany": ("Berlin", "Europe", "the euro", "German"),
    "Italy": ("Rome", "Europe", "the euro", "Italian"),
    "Spain": ("Madrid", "Europe", "the euro", "Spanish"),
    "Portugal": ("Lisbon", "Europe", "the euro", "Portuguese"),
    "the United Kingdom": ("London", "Europe", "the pound sterling", "English"),
    "Ireland": ("Dublin", "Europe", "the euro", "English and Irish"),
    "the Netherlands": ("Amsterdam", "Europe", "the euro", "Dutch"),
    "Belgium": ("Brussels", "Europe", "the euro", "Dutch, French and German"),
    "Switzerland": ("Bern", "Europe", "the Swiss franc", "German, French, Italian and Romansh"),
    "Austria": ("Vienna", "Europe", "the euro", "German"),
    "Sweden": ("Stockholm", "Europe", "the Swedish krona", "Swedish"),
    "Norway": ("Oslo", "Europe", "the Norwegian krone", "Norwegian"),
    "Denmark": ("Copenhagen", "Europe", "the Danish krone", "Danish"),
    "Finland": ("Helsinki", "Europe", "the euro", "Finnish and Swedish"),
    "Poland": ("Warsaw", "Europe", "the Polish zloty", "Polish"),
    "Greece": ("Athens", "Europe", "the euro", "Greek"),
    "Russia": ("Moscow", "Europe and Asia", "the Russian ruble", "Russian"),
    "Ukraine": ("Kyiv", "Europe", "the Ukrainian hryvnia", "Ukrainian"),
    "Turkey": ("Ankara", "Europe and Asia", "the Turkish lira", "Turkish"),
    "India": ("New Delhi", "Asia", "the Indian rupee", "Hindi and English, along with many regional languages"),
    "China": ("Beijing", "Asia", "the renminbi (yuan)", "Mandarin Chinese"),
    "Japan": ("Tokyo", "Asia", "the Japanese yen", "Japanese"),
    "South Korea": ("Seoul", "Asia", "the South Korean won", "Korean"),
    "Thailand": ("Bangkok", "Asia", "the Thai baht", "Thai"),
    "Vietnam": ("Hanoi", "Asia", "the Vietnamese dong", "Vietnamese"),
    "Malaysia": ("Kuala Lumpur", "Asia", "the Malaysian ringgit", "Malay"),
    "Singapore": ("Singapore", "Asia", "the Singapore dollar", "English, Malay, Mandarin and Tamil"),
    "the Philippines": ("Manila", "Asia", "the Philippine peso", "Filipino and English"),
    "Pakistan": ("Islamabad", "Asia", "the Pakistani rupee", "Urdu and English"),
    "Bangladesh": ("Dhaka", "Asia", "the Bangladeshi taka", "Bengali"),
    "Nepal": ("Kathmandu", "Asia", "the Nepalese rupee", "Nepali"),
    "Saudi Arabia": ("Riyadh", "Asia", "the Saudi riyal", "Arabic"),
    "the United Arab Emirates": ("Abu Dhabi", "Asia", "the UAE dirham", "Arabic"),
    "Iran": ("Tehran", "Asia", "the Iranian rial", "Persian"),
    "Egypt": ("Cairo", "Africa", "the Egyptian pound", "Arabic"),
    "Nigeria": ("Abuja", "Africa", "the Nigerian naira", "English"),
    "Kenya": ("Nairobi", "Africa", "the Kenyan shilling", "Swahili and English"),
    "Ethiopia": ("Addis Ababa", "Africa", "the Ethiopian birr", "Amharic"),
    "Morocco": ("Rabat", "Africa", "the Moroccan dirham", "Arabic and Berber"),
    "Ghana": ("Accra", "Africa", "the Ghanaian cedi", "English"),
    "the United States": ("Washington, D.C.", "North America", "the US dollar", "English"),
    "Canada": ("Ottawa", "North America", "the Canadian dollar", "English and French"),
    "Mexico": ("Mexico City", "North America", "the Mexican peso", "Spanish"),
    "Brazil": ("Brasília", "South America", "the Brazilian real", "Portuguese"),
    "Argentina": ("Buenos Aires", "South America", "the Argentine peso", "Spanish"),
    "Chile": ("Santiago", "South America", "the Chilean peso", "Spanish"),
    "Peru": ("Lima", "South America", "the Peruvian sol", "Spanish"),
    "Colombia": ("Bogotá", "South America", "the Colombian peso", "Spanish"),
    "Australia": ("Canberra", "Oceania", "the Australian dollar", "English"),
    "New Zealand": ("Wellington", "Oceania", "the New Zealand dollar", "English and Maori"),
}

# element: (symbol, atomic number, short note)
ELEMENTS = {
    "hydrogen": ("H", 1, "the lightest and most common element in the universe"),
    "helium": ("He", 2, "a light noble gas used in balloons"),
    "lithium": ("Li", 3, "a light metal used in rechargeable batteries"),
    "carbon": ("C", 6, "the basis of all known life"),
    "nitrogen": ("N", 7, "about 78% of Earth's air"),
    "oxygen": ("O", 8, "the gas we need to breathe, about 21% of Earth's air"),
    "sodium": ("Na", 11, "a soft metal; with chlorine it makes table salt"),
    "magnesium": ("Mg", 12, "a light metal that burns with a bright white flame"),
    "aluminium": ("Al", 13, "a light metal used in cans and aircraft"),
    "silicon": ("Si", 14, "the material computer chips are made from"),
    "sulfur": ("S", 16, "a yellow element that smells like rotten eggs in some compounds"),
    "chlorine": ("Cl", 17, "a gas used to disinfect water"),
    "potassium": ("K", 19, "a metal that's important for nerves and muscles"),
    "calcium": ("Ca", 20, "the main mineral in bones and teeth"),
    "iron": ("Fe", 26, "the metal used to make steel"),
    "copper": ("Cu", 29, "a metal that conducts electricity very well"),
    "zinc": ("Zn", 30, "a metal used to protect steel from rust"),
    "silver": ("Ag", 47, "the best conductor of electricity of all metals"),
    "tin": ("Sn", 50, "a metal used in solder and tin cans"),
    "iodine": ("I", 53, "an element your thyroid needs"),
    "gold": ("Au", 79, "a precious metal that doesn't rust"),
    "mercury": ("Hg", 80, "the only metal that is liquid at room temperature"),
    "lead": ("Pb", 82, "a heavy, toxic metal"),
    "uranium": ("U", 92, "a radioactive element used as nuclear fuel"),
}

PLANETS = [
    ("Mercury", "the closest planet to the Sun and the smallest planet in the solar system"),
    ("Venus", "the second planet from the Sun and the hottest, because its thick clouds trap heat"),
    ("Earth", "the third planet from the Sun and the only one known to have life"),
    ("Mars", "the fourth planet, called the Red Planet because of iron oxide (rust) in its soil; it has two small moons, Phobos and Deimos"),
    ("Jupiter", "the fifth planet and the largest in the solar system, a gas giant with a huge storm called the Great Red Spot"),
    ("Saturn", "the sixth planet, a gas giant famous for its bright rings of ice and rock"),
    ("Uranus", "the seventh planet, an ice giant that spins on its side"),
    ("Neptune", "the eighth and farthest planet from the Sun, an ice giant with the fastest winds in the solar system"),
]

# (question topic, who, answer)
INVENTIONS = [
    ("the telephone", "Alexander Graham Bell", "Alexander Graham Bell received the first patent for the telephone in 1876."),
    ("the light bulb", "Thomas Edison", "Thomas Edison made the first practical, long-lasting light bulb in 1879, building on earlier inventors' work."),
    ("the World Wide Web", "Tim Berners-Lee", "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN."),
    ("Python", "Guido van Rossum", "Guido van Rossum created Python and released it in 1991."),
    ("JavaScript", "Brendan Eich", "Brendan Eich created JavaScript in 1995 at Netscape, in just about ten days."),
    ("Linux", "Linus Torvalds", "Linus Torvalds created the Linux kernel in 1991."),
    ("the C programming language", "Dennis Ritchie", "Dennis Ritchie created C at Bell Labs in the early 1970s."),
    ("the printing press", "Johannes Gutenberg", "Johannes Gutenberg invented the movable-type printing press in Europe around 1440."),
    ("the airplane", "the Wright brothers", "Orville and Wilbur Wright made the first powered airplane flight in 1903."),
    ("penicillin", "Alexander Fleming", "Alexander Fleming discovered penicillin in 1928."),
    ("the theory of relativity", "Albert Einstein", "Albert Einstein developed the theory of relativity: special relativity in 1905 and general relativity in 1915."),
    ("the laws of motion and gravity", "Isaac Newton", "Isaac Newton described the laws of motion and universal gravitation in 1687."),
    ("the theory of evolution by natural selection", "Charles Darwin", "Charles Darwin described evolution by natural selection in 'On the Origin of Species' in 1859."),
    ("the first computer program", "Ada Lovelace", "Ada Lovelace wrote what is considered the first computer program, for Charles Babbage's Analytical Engine, in the 1840s."),
    ("the Turing machine", "Alan Turing", "Alan Turing described the Turing machine in 1936, laying the foundations of computer science."),
    ("the radio", "Guglielmo Marconi", "Guglielmo Marconi developed the first practical radio communication system in the 1890s."),
    ("the steam engine", "James Watt", "James Watt greatly improved the steam engine in the 1760s and 1770s, helping power the Industrial Revolution."),
]

# Single facts: (questions, answer)
FACTS = [
    (["What is the largest ocean?", "Which ocean is the biggest?"],
     "The Pacific Ocean is the largest and deepest ocean on Earth."),
    (["What is the tallest mountain in the world?", "How tall is Mount Everest?"],
     "Mount Everest is the tallest mountain above sea level, at about 8,849 meters (29,032 feet)."),
    (["What is the longest river in the world?", "Which river is the longest?"],
     "The Nile is usually listed as the longest river, at about 6,650 km, although some measurements put the Amazon slightly longer."),
    (["What is the largest desert?", "Which is the biggest desert?"],
     "Antarctica is the largest desert overall. The largest hot desert is the Sahara in Africa."),
    (["What is the largest country in the world?", "Which country is the biggest?"],
     "Russia is the largest country by area, covering about 17 million square kilometers."),
    (["What is the most populous country?", "Which country has the most people?"],
     "India and China are the two most populous countries, each with more than 1.4 billion people. India overtook China in 2023."),
    (["How many continents are there?", "Name the continents"],
     "There are seven continents: Africa, Antarctica, Asia, Australia (Oceania), Europe, North America and South America."),
    (["How many oceans are there?", "Name the oceans"],
     "There are five oceans: the Pacific, Atlantic, Indian, Southern and Arctic Oceans."),
    (["How many planets are in the solar system?", "Name the planets"],
     "There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune."),
    (["What is the speed of light?", "How fast is light?"],
     "Light travels at about 299,792 kilometers per second in a vacuum."),
    (["How far is the Moon from Earth?", "What is the distance to the Moon?"],
     "The Moon is about 384,400 kilometers from Earth on average."),
    (["How far is the Sun from Earth?", "What is the distance to the Sun?"],
     "The Sun is about 150 million kilometers from Earth. Its light takes about 8 minutes to reach us."),
    (["How old is the Earth?", "What is the age of the Earth?"],
     "The Earth is about 4.5 billion years old."),
    (["How old is the universe?", "What is the age of the universe?"],
     "The universe is about 13.8 billion years old."),
    (["Who was the first person on the Moon?", "When did humans land on the Moon?"],
     "Neil Armstrong was the first person to walk on the Moon, on July 20, 1969, during NASA's Apollo 11 mission."),
    (["Who was the first person in space?", "Who first went to space?"],
     "Yuri Gagarin of the Soviet Union was the first person in space, on April 12, 1961."),
    (["At what temperature does water boil?", "What is the boiling point of water?"],
     "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level."),
    (["At what temperature does water freeze?", "What is the freezing point of water?"],
     "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit)."),
    (["How do I convert Celsius to Fahrenheit?", "Celsius to Fahrenheit formula"],
     "Multiply by 9/5 and add 32: F = C x 9/5 + 32. For example, 25 C is 77 F."),
    (["How do I convert Fahrenheit to Celsius?", "Fahrenheit to Celsius formula"],
     "Subtract 32 and multiply by 5/9: C = (F - 32) x 5/9. For example, 98.6 F is 37 C."),
    (["How many kilometers are in a mile?", "Convert miles to kilometers"],
     "One mile is about 1.609 kilometers, and one kilometer is about 0.621 miles."),
    (["How many centimeters are in an inch?", "Convert inches to centimeters"],
     "One inch is exactly 2.54 centimeters."),
    (["How many pounds are in a kilogram?", "Convert kilograms to pounds"],
     "One kilogram is about 2.205 pounds, and one pound is about 0.454 kilograms."),
    (["How many bones are in the human body?", "How many bones does an adult have?"],
     "An adult human has 206 bones. Babies are born with around 270, and some fuse together as they grow."),
    (["How many chambers does the heart have?", "How does the heart work?"],
     "The human heart has four chambers: two atria on top and two ventricles below. It pumps blood through the lungs to collect oxygen and then around the body."),
    (["What is normal body temperature?", "What is a normal human temperature?"],
     "Normal body temperature is about 37 degrees Celsius (98.6 degrees Fahrenheit), though it varies a little from person to person."),
    (["What is the largest organ in the human body?", "Which organ is the biggest?"],
     "The skin is the largest organ of the human body."),
    (["What is the largest animal?", "What is the biggest animal ever?"],
     "The blue whale is the largest animal ever known. It can grow to about 30 meters long."),
    (["What is the fastest land animal?", "Which animal runs the fastest?"],
     "The cheetah is the fastest land animal, reaching about 100 to 120 km/h in short sprints."),
    (["How many days are in a year?", "How long is a year?"],
     "A year has 365 days, or 366 in a leap year. It's the time Earth takes to orbit the Sun."),
    (["What is a leap year?", "Why do we have leap years?"],
     "A leap year has an extra day, February 29. It happens almost every four years to keep the calendar in step with Earth's orbit, which takes about 365.24 days."),
    (["How many seconds are in a day?", "How many minutes are in a day?"],
     "A day has 24 hours, 1,440 minutes and 86,400 seconds."),
    (["What is the value of pi?", "What is pi?"],
     "Pi is the ratio of a circle's circumference to its diameter. It's about 3.14159 and its digits go on forever without repeating."),
    (["What is the Pythagorean theorem?", "Explain Pythagoras theorem"],
     "In a right-angled triangle, the square of the longest side equals the sum of the squares of the other two sides: a^2 + b^2 = c^2."),
    (["What is a prime number?", "Define prime number"],
     "A prime number is a whole number greater than 1 that can only be divided evenly by 1 and itself, like 2, 3, 5, 7 and 11."),
    (["What is the tallest building in the world?", "Which building is the tallest?"],
     "As of my last update, the Burj Khalifa in Dubai is the tallest building in the world, at about 828 meters."),
    (["What is the United Nations?", "What does the UN do?"],
     "The United Nations is an international organization founded in 1945. It has 193 member states and works on peace, security, human rights and development."),
    (["When did World War II end?", "When was World War 2?"],
     "World War II lasted from 1939 to 1945. It ended in Europe in May 1945 and in Asia in September 1945."),
    (["When did World War I happen?", "When was World War 1?"],
     "World War I lasted from 1914 to 1918."),
    (["When did India become independent?", "When is India's Independence Day?"],
     "India became independent from British rule on August 15, 1947, which is celebrated as Independence Day."),
    (["When did the United States become independent?", "When was the Declaration of Independence signed?"],
     "The United States declared independence on July 4, 1776."),
    (["Who wrote Romeo and Juliet?", "Who was William Shakespeare?"],
     "William Shakespeare wrote Romeo and Juliet. He was an English playwright and poet who lived from 1564 to 1616."),
    (["Who painted the Mona Lisa?", "Who was Leonardo da Vinci?"],
     "Leonardo da Vinci painted the Mona Lisa in the early 1500s. He was an Italian Renaissance artist, scientist and inventor."),
    (["What is the Great Wall of China?", "How long is the Great Wall of China?"],
     "The Great Wall of China is a series of ancient fortifications built over many centuries. All its sections together stretch more than 20,000 km."),
    (["What is the Taj Mahal?", "Where is the Taj Mahal?"],
     "The Taj Mahal is a white marble mausoleum in Agra, India, built by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal. It was completed around 1653."),
    (["What are the colors of the rainbow?", "How many colors are in a rainbow?"],
     "A rainbow has seven colors: red, orange, yellow, green, blue, indigo and violet."),
    (["Why is the sky blue?", "What makes the sky blue?"],
     "Sunlight is scattered by air molecules, and blue light is scattered much more than red light because of its shorter wavelength. That scattered blue light fills the sky."),
    (["What is the chemical formula of water?", "What is water made of?"],
     "Water is H2O: each molecule has two hydrogen atoms and one oxygen atom."),
    (["What is the chemical formula for table salt?", "What is salt made of?"],
     "Table salt is sodium chloride, NaCl."),
    (["What gas do plants absorb?", "What do plants need to grow?"],
     "Plants absorb carbon dioxide from the air and use sunlight and water to make sugar through photosynthesis, releasing oxygen."),
]


def _capital(country, cap, cont, cur, lang):
    C = country[0].upper() + country[1:]
    yield [f"What is the capital of {country}?"], f"The capital of {country} is {cap}."
    yield [f"capital of {country}"], f"{cap} is the capital of {country}."
    yield [f"Which continent is {country} in?"], f"{C} is in {cont}."
    yield [f"What currency does {country} use?"], f"{C} uses {cur}."
    yield [f"What language is spoken in {country}?"], f"The main language of {country} is {lang}." if " and " not in lang and "," not in lang else f"The main languages of {country} are {lang}."
    yield [f"Tell me about {country}"], f"{C} is a country in {cont}. Its capital is {cap}, its currency is {cur}, and its main language{'s are' if ' and ' in lang or ',' in lang else ' is'} {lang}."


def conversations():
    convs = []
    for country, info in COUNTRIES.items():
        convs += list(_capital(country, *info))
    for name, (sym, num, note) in ELEMENTS.items():
        convs.append(([f"What is the chemical symbol for {name}?"], f"The chemical symbol for {name} is {sym}."))
        convs.append(([f"What is {name}?"], f"{name.capitalize()} ({sym}) is element number {num} on the periodic table: {note}."))
        convs.append(([f"What is the atomic number of {name}?"], f"The atomic number of {name} is {num}."))
    for i, (planet, desc) in enumerate(PLANETS, 1):
        convs.append(([f"Tell me about {planet}", f"What is {planet}?"], f"{planet} is {desc}."))
    for topic, who, answer in INVENTIONS:
        convs.append(([f"Who invented {topic}?", f"Who created {topic}?"], answer))
        convs.append(([f"Who is {who}?"] if not who.startswith("the ") else [f"Who were {who}?"], answer))
    convs += FACTS
    return convs


def write(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Generated by `python -m maxgpt.knowledge` from maxgpt/knowledge.py. Edit that file, not this one.\n")
        for questions, answer in conversations():
            for q in questions:
                f.write(f"===\nUser: {q}\nMax: {answer}\n")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chat", "20_world_facts.txt")
    write(out)
    print(f"wrote {out}")
