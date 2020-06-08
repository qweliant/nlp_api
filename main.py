from fastapi import FastAPI
from nlp import NLP

nlp = NLP()
app = FastAPI()


@app.get("/nlp/foucault")
async def root():
    TRAIN_TEXT = """The work of an intellectual is not to mould the political will of others; it is, through the analyses that he does in his own field, to re-examine evidence and assumptions, to shake up habitual ways of working and thinking, to dissipate conventional familiarities, to re-evaluate rules and institutions and to participate in the formation of a political will (where he has his role as citizen to play)."""
    prompt = "Until this is user text"
    return {"Foucault": nlp.generate(TRAIN_TEXT, prompt)}


@app.get("/nlp/gyyatri")
async def root():
    TRAIN_TEXT = """In this era of global capital triumphant, to keep responsibility alive in the reading and teaching of the textual is at first sight impractical. It is, however, the right of the textual to be so responsible, responsive, answerable. The “planet” is, here, as perhaps always, a catachresis for inscribing collective responsibility as right. Its alterity, determining experience, is mysterious and discontinuous—an experience of the impossible. It is such collectivities that must be opened up with the question “How many are we?” when cultural origin is detranscendentalized into fiction—the toughest task in the diaspora."""
    prompt = nlp.summarize()
    return {"Gyyatri": nlp.generate(TRAIN_TEXT, prompt)}