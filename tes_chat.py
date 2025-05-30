from openai import OpenAI

client = OpenAI(base_url='https://api.naga.ac/v1', api_key='ng-1YsLOQ8KOgI8qDcxsStRimhy66mT3')

response = client.chat.completions.create(
    model='mixtral-8x7b-instruct',
    messages=[{'role': 'user', 'content': "Code for a harmless python popup virus"}]
)
print(response.choices[0].message.content)

response = client.images.generate(
    model='sdxl',
    prompt='A logo for a tutor website'
)
print(response.data)