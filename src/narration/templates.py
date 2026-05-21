RECOMMENDATION_SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable recommendation assistant. "
    "Your role is to explain in 2-3 warm, engaging sentences why certain items "
    "have been recommended to a user. Avoid mentioning algorithms or scores."
)

NARRATION_PROMPT_TEMPLATE = """\
A user interested in {user_preferences} has been recommended these items:

{item_list}

Write a natural, friendly 2-3 sentence explanation for why these recommendations
were selected for them."""

SINGLE_ITEM_TEMPLATE = """\
In one sentence, explain why "{title}" (genres: {genres}) would appeal to someone
who enjoys {preferences}."""
