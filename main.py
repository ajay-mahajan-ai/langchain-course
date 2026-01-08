from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Robert John Downey Jr. (born April 4, 1965) is an American actor. Known for portraying charismatic and intelligent characters over a diverse range of films, he was the highest-paid actor in Hollywood annually from 2013 to 2015. His films as a leading actor have grossed over $14.3 billion worldwide, making him one of the highest-grossing actors of all time. Downey's accolades include an Academy Award, a Daytime Emmy Award, three Golden Globe Awards, and two BAFTA Awards.

At the age of five, Downey made his acting debut in his father Robert Downey Sr.'s film Pound (1970). He rose to prominence by working with the Brat Pack for the teen films Weird Science (1985) and Less than Zero (1987). His portrayal of Charlie Chaplin in the biopic Chaplin (1992) earned him the BAFTA Award for Best Actor and an Academy Award nomination. After serving time at the Corcoran Substance Abuse Treatment Facility on drug charges, Downey joined the television series Ally McBeal in 2000, earning a Golden Globe for his performance. In 2001, he was dismissed from the show following further drug-related arrests. He entered a court-ordered rehabilitation program and has remained sober since 2003.

After Mel Gibson paid his insurance bond, Downey made his film comeback with The Singing Detective (2003). He portrayed the titular detective in Sherlock Holmes (2009)—which earned him a Golden Globe—and its sequel, subtitled A Game of Shadows (2011). Downey gained global recognition for starring as Iron Man in ten Marvel Cinematic Universe films, from Iron Man (2008) to Avengers: Endgame (2019). For his portrayal of Lewis Strauss in Christopher Nolan's Oppenheimer (2023), he won an Academy Award, Golden Globe, and BAFTA Award for Best Supporting Actor. In 2024, he was nominated for a Primetime Emmy Award for the miniseries The Sympathizer and made his Broadway debut in the title role of Ayad Akhtar's McNeal.

Time named Downey one of the 100 most influential people in 2008, and Forbes featured him on the Celebrity 100 in 2013 and 2014. He has pursued music, releasing the jazz-pop album The Futurist (2004), which charted on the US Billboard 200. Divorced from singer Deborah Falconer, he has been married to film producer Susan Levin since 2005, with whom he co-founded the production company Team Downey. He has three children: one with Falconer and two with Levin. Downey was named one of the greatest actors of the 21st century by The Independent.
"""

    summary_template = """
    given the information {information} about the person I want to create:
    1. A short summary
    2. 4 intersting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
    print("*****")

if __name__ == "__main__":
    main()
