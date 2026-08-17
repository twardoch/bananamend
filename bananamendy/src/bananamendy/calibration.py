# this_file: bananamendy/src/bananamendy/calibration.py
"""Text for the calibration of a quantizer, and text for the measurement.

A quantizer that knows the inputs of each matrix makes better decisions. The
text below gives those inputs. It is in the package, so the work needs no
network and always uses the same words.

The two sets are separate on purpose. If the measurement used the calibration
text, a good number could mean only that the quantizer learned that text.

The subjects are ordinary technical prose, questions and answers, lists, numbers
and short conversations, because a checkpoint of this family sees that kind of
text. The text is in English, and a checkpoint for another language needs its own
calibration text.
"""

from __future__ import annotations

CALIBRATION_TEXTS: tuple[str, ...] = (
    """A river carries water from the hills to the sea. The water moves stones, and the
stones make the bed of the river smooth. A mill on the bank turns a wheel, and the
wheel turns a stone that grinds the wheat into flour. The miller opens a gate to
control the flow. When the gate is open, the wheel turns quickly. When the gate is
closed, the wheel stops. The mill worked for two hundred years, and then a machine
with a motor did the same work in one hour.""",
    """Steel is an alloy of iron and carbon. More carbon makes the steel harder, and it
also makes the steel easier to break. A blacksmith heats the steel until it is red,
hits it into shape, and then cools it in oil or in water. Fast cooling makes the
steel hard. Slow cooling makes the steel soft. The blacksmith heats the steel again
at a lower temperature to remove some of the hardness, because a tool that is too
hard breaks in use.""",
    """The engineer measures the bridge every winter. Water enters a small crack. The
water freezes, and ice takes more space than water, so the crack becomes wider. A
wide crack holds more water, and the next winter makes it wider again. The repair
is simple if the engineer finds the crack early: clean the crack, fill it, and seal
the surface. The repair is expensive if the crack reaches the steel inside the
concrete.""",
    """Question: why does a ship of steel float?
Answer: the ship pushes away a mass of water that is larger than the mass of the
ship. The shape of the hull does the work, and not the metal. A solid block of
steel sinks, because it pushes away only a small mass of water.
Question: what happens if the hull has a hole?
Answer: water enters, the mass of the ship increases, and the ship goes down until
the mass of the water that it pushes away equals its own mass again. If the hole is
large, the ship sinks.""",
    """In the morning the baker weighs the flour, the salt, the water and the yeast. The
dough rests for one hour at twenty degrees. The baker folds the dough two times,
and the dough rests again. The oven is at two hundred and thirty degrees. The bread
bakes for thirty minutes, and then it cools on a rack for one hour. Bread that is
cut when it is hot loses its shape.""",
    """The capital of France is Paris. The capital of Poland is Warsaw. The capital of
Germany is Berlin. The capital of Japan is Tokyo. The Pacific is the largest ocean,
and the Atlantic is the second largest. The Nile and the Amazon are long rivers.
Mount Everest is the highest mountain above the sea. The Dead Sea is the lowest
water on land.""",
    """Water freezes at zero degrees Celsius, and it boils at one hundred degrees at the
pressure of the sea. At a high altitude the pressure is lower, so water boils at a
lower temperature, and food needs more time to cook. Ice is less dense than water,
so ice floats. Most solids sink in their own liquid, and water is unusual. The
reason is the shape of the water molecule and the bonds between the molecules.""",
    """A safe kitchen has these rules. Keep the knives sharp, because a blunt knife slips.
Keep the floor dry, because a wet floor is a fall. Turn the handles of the pans
inwards, because a handle above the edge is a burn. Keep a lid near the stove: a lid
stops a fire in a pan, and water makes that fire worse. Wash the hands before the
work, and again after raw meat.""",
    """Step 1. Switch off the power at the box.
Step 2. Make sure that the power is off with a test lamp.
Step 3. Open the case, and take a photograph of the wires.
Step 4. Remove the old switch.
Step 5. Connect the new switch. The colours must agree with the photograph.
Step 6. Close the case, switch on the power, and test the light two times.""",
    """The library opens at nine in the morning and closes at six in the evening. On
Saturday it closes at one. It is closed on Sunday. A reader can borrow six books for
three weeks. A late book costs ten cents for each day. The reading room on the
second floor is quiet, and food is not permitted there.""",
    """A computer has a processor, a memory and a disk. The processor does the work, the
memory holds the data of the current work, and the disk keeps the data when the
power is off. The memory is fast and small. The disk is slow and large. A program
that reads the disk many times is slow, so a good program keeps the data that it
needs in the memory.""",
    """The doctor asked three questions. When did the pain start? Where is the pain? Does
the pain move? The answers were: two days ago, in the right side, and no. The doctor
measured the temperature and the pressure of the blood. The temperature was high.
The doctor sent the patient to the hospital for a photograph of the inside of the
body.""",
    """Dear Sir or Madam,
I write about the invoice of the fourth of March. The invoice shows two items, and
we received only one. Please send the second item, or send a new invoice for one
item. Our order number is 4471.
Yours faithfully,
A. Kowalski""",
    """The train leaves platform three at 07:42 and arrives at 09:15. The ticket costs
fourteen euros. A reservation is not necessary, but a seat is not certain in the
morning. The train stops at four stations. Coffee is available in the second
carriage. A bicycle needs a separate ticket of three euros.""",
    """A plant needs light, water, air and food from the soil. Too much water removes the
air from the soil, and then the roots die. Too little water closes the small holes
in the leaves, and then the plant stops growing. The gardener puts a finger in the
soil: if the soil is dry two centimetres below the surface, the plant needs water.""",
    """Machine translation was a difficult problem for fifty years. The early systems used
rules, and a rule that is correct for one sentence is wrong for the next sentence.
The later systems used pairs of sentences and statistics. The current systems use a
network with many layers, and they learn from a very large number of sentences. The
result is good, and it is still not perfect.""",
    """The teacher wrote three numbers on the board: 12, 18 and 30. What is the largest
number that divides all three? The answer is 6. What is the smallest number that all
three divide? The answer is 180. A student asked why the two answers are so far
apart. The teacher drew the factors of each number and showed the shared part and
the complete part.""",
    """A good measurement has three properties. It is repeatable: the same work gives the
same number. It is comparable: two people with the same instrument get the same
number. It is honest: the report includes the error. A measurement without an error
is not a measurement; it is an opinion with a number in it.""",
    """The hotel has forty rooms. Twelve rooms look at the sea, and the others look at the
garden. Breakfast is from seven to ten. The kitchen serves food until nine in the
evening. The nearest station is one kilometre away, and a bus stops in front of the
hotel every twenty minutes. A room with a view of the sea costs thirty euros more.""",
    """Never put water on burning oil. The water becomes steam in one moment, the steam
throws the oil into the air, and the fire becomes much larger. Close the pan with a
lid, and switch off the heat. If the fire does not stop, leave the room, close the
door, and call for help. Do not open the door again to look.""",
)

# The measurement uses this text. The quantizer never sees it.
EVALUATION_TEXT = """
The old mill stood at the bend of the river. For two centuries it ground the wheat
of three villages. People said that the miller could read the weather in the sound
of the water. When the river ran high, he closed the gate early. When it whispered,
he let the stones turn through the night.

Ice floats because ice is less dense than water. Most solids sink in their own
liquid, and water is unusual. A lake therefore freezes at the surface, and the fish
below the ice stay alive through the winter.

A safe workshop has three rules. Put every tool back in its place. Keep the light
above the work, and not behind it. Stop when you are tired, because a tired person
makes the same mistake two times.

The capital of Italy is Rome. The largest ocean is the Pacific. A whale is a mammal,
and a shark is a fish. Water boils at one hundred degrees at the pressure of the sea.
""".strip()

EVALUATION_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "Water freezes at",
    "The largest ocean is",
    "One, two, three,",
    "A safe kitchen has",
)

EVALUATION_CHATS: tuple[tuple[dict[str, str], ...], ...] = (
    ({"role": "user", "content": "Name one ocean."},),
    ({"role": "user", "content": "Why does ice float?"},),
    ({"role": "user", "content": "Give one rule for a safe kitchen."},),
)
