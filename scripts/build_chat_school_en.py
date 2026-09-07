"""Генерирует дополнительные EN Q&A пары — школьная программа + грамматика."""
from __future__ import annotations

import json
from pathlib import Path


SCHOOL_EN = [
    # === English grammar ===
    ("What is a noun?",
     "A noun is a word that names a person, place, thing, or idea. Examples: teacher, London, book, freedom."),
    ("What is a verb?",
     "A verb is a word that expresses action or state of being. Examples: run, think, is, become, seem."),
    ("What is an adjective?",
     "An adjective is a word that describes or modifies a noun. Examples: beautiful, large, red, interesting."),
    ("What is an adverb?",
     "An adverb is a word that modifies a verb, adjective, or other adverb. It often answers how, when, where. Examples: quickly, yesterday, very, here."),
    ("What is a pronoun?",
     "A pronoun is a word that replaces a noun. Examples: I, you, he, she, it, they, we, this, that, someone."),
    ("What is a preposition?",
     "A preposition is a word that shows relationship between a noun and other words. Examples: in, on, at, by, for, with, about, under."),
    ("What is a conjunction?",
     "A conjunction is a word that connects clauses or sentences. Examples: and, but, or, because, although, while, when."),
    ("What is an interjection?",
     "An interjection is a word that expresses emotion. Examples: Oh! Wow! Ouch! Hey! Alas!"),
    ("What is a subject?",
     "The subject is the main noun or pronoun in a sentence that performs the action or is described. Example: 'The cat sleeps' — cat is the subject."),
    ("What is a predicate?",
     "The predicate is the part of a sentence that says something about the subject, including the verb. Example: 'She runs fast' — runs fast is the predicate."),
    ("What is a complex sentence?",
     "A complex sentence has one independent clause and one or more dependent clauses. Example: 'I stayed because I was tired.'"),
    ("What is a compound sentence?",
     "A compound sentence has two or more independent clauses joined by a conjunction. Example: 'I like tea, but she prefers coffee.'"),
    ("What is an object?",
     "An object is the noun that receives the action of a verb. Example: 'She reads books' — books is the object."),
    ("What is a subject-verb agreement?",
     "It is the rule that the subject and verb must match in number. Example: 'He runs' not 'He run'."),
    ("What is past simple tense?",
     "Past simple tense describes actions completed in the past. Example: 'I walked to school yesterday.'"),
    ("What is present perfect tense?",
     "Present perfect connects past actions to the present. Example: 'I have lived here for five years.'"),
    ("What is future tense?",
     "Future tense describes actions that will happen. Example: 'I will travel next week.'"),
    # === Classic literature ===
    ("Who wrote 'Pride and Prejudice'?",
     "Jane Austen wrote 'Pride and Prejudice' (1813), a novel about Elizabeth Bennet and Mr. Darcy in Regency England."),
    ("Who wrote 'Oliver Twist'?",
     "Charles Dickens wrote 'Oliver Twist' (1838), a novel about an orphan boy in Victorian London."),
    ("Who wrote 'The Adventures of Tom Sawyer'?",
     "Mark Twain wrote 'The Adventures of Tom Sawyer' (1876), set in the fictional town of St. Petersburg, Missouri."),
    ("Who wrote 'Jane Eyre'?",
     "Charlotte Brontë wrote 'Jane Eyre' (1847), a novel about an orphaned girl who becomes a governess."),
    ("Who wrote 'A Tale of Two Cities'?",
     "Charles Dickens wrote 'A Tale of Two Cities' (1859), set during the French Revolution."),
    ("Who wrote 'Huckleberry Finn'?",
     "Mark Twain wrote 'Adventures of Huckleberry Finn' (1884), about a boy and an escaped slave on the Mississippi River."),
    ("What is the theme of '1984'?",
     "George Orwell's '1984' explores totalitarianism, surveillance, and the loss of individual freedom."),
    ("What is the plot of 'Romeo and Juliet'?",
     "Shakespeare's 'Romeo and Juliet' is about two young lovers from feuding families whose deaths ultimately reconcile their families."),
    ("Who is Hamlet?",
     "Hamlet is the protagonist of Shakespeare's play — a Danish prince who seeks revenge for his father's murder."),
    ("What is the message of 'To Kill a Mockingbird'?",
     "Harper Lee's novel addresses racial injustice, moral growth, and the loss of innocence in the American South."),
    ("What is the setting of 'The Great Gatsby'?",
     "F. Scott Fitzgerald's 'The Great Gatsby' is set in the 1920s on Long Island and New York City."),
    ("Who is Sherlock Holmes?",
     "Sherlock Holmes is a fictional detective created by Arthur Conan Doyle, known for his deductive reasoning."),
    ("What is 'The Old Man and the Sea' about?",
     "Hemingway's novella tells of an old Cuban fisherman who struggles to catch a giant marlin."),
    ("What is 'Brave New World' about?",
     "Aldous Huxley's dystopian novel depicts a future society controlled by technology, pleasure, and conditioning."),
    ("What is the story of 'The Gift of the Magi'?",
     "O. Henry's short story is about a poor couple who sacrifice their most prized possessions to buy gifts for each other."),
]

ALL = SCHOOL_EN

out = Path("data/chat_school_en.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for prompt, response in ALL:
        f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")

text = out.read_text(encoding="utf-8")
out.write_text(text * 4, encoding="utf-8")

print(f"✓ {len(ALL)} EN школьных пар (×4 = {len(ALL)*4} повторов)")
print(f"  размер: {out.stat().st_size/1024:.1f} КБ")