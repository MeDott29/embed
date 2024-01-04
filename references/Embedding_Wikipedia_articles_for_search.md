# Embedding Wikipedia articles for search

This notebook shows how we prepared a dataset of Wikipedia articles for search, used in [Question_answering_using_embeddings.ipynb](Question_answering_using_embeddings.ipynb).

Procedure:

0. Prerequisites: Import libraries, set API key (if needed)
1. Collect: We download a few hundred Wikipedia articles about the 2022 Olympics
2. Chunk: Documents are split into short, semi-self-contained sections to be embedded
3. Embed: Each section is embedded with the OpenAI API
4. Store: Embeddings are saved in a CSV file (for large datasets, use a vector database)

## 0. Prerequisites

### Import libraries


```python
# imports
import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens

```

Install any missing libraries with `pip install` in your terminal. E.g.,

```zsh
pip install openai
```

(You can also do this in a notebook cell with `!pip install openai`.)

If you install any libraries, be sure to restart the notebook kernel.

### Set API key (if needed)

Note that the OpenAI library will try to read your API key from the `OPENAI_API_KEY` environment variable. If you haven't already, set this environment variable by following [these instructions](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

## 1. Collect documents

In this example, we'll download a few hundred Wikipedia articles related to the 2022 Winter Olympics.


```python
# get Wikipedia pages about the 2022 Winter Olympics

CATEGORY_TITLE = "Category:2022 Winter Olympics"
WIKI_SITE = "en.wikipedia.org"


def titles_from_category(
    category: mwclient.listing.Category, max_depth: int
) -> set[str]:
    """Return a set of page titles in a given Wiki category and its subcategories."""
    titles = set()
    for cm in category.members():
        if type(cm) == mwclient.page.Page:
            # ^type() used instead of isinstance() to catch match w/ no inheritance
            titles.add(cm.name)
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)
            titles.update(deeper_titles)
    return titles


site = mwclient.Site(WIKI_SITE)
category_page = site.pages[CATEGORY_TITLE]
titles = titles_from_category(category_page, max_depth=1)
# ^note: max_depth=1 means we go one level deep in the category tree
print(f"Found {len(titles)} article titles in {CATEGORY_TITLE}.")

```

    Found 731 article titles in Category:2022 Winter Olympics.


## 2. Chunk documents

Now that we have our reference documents, we need to prepare them for search.

Because GPT can only read a limited amount of text at once, we'll split each document into chunks short enough to be read.

For this specific example on Wikipedia articles, we'll:
- Discard less relevant-looking sections like External Links and Footnotes
- Clean up the text by removing reference tags (e.g., <ref>), whitespace, and super short sections
- Split each article into sections
- Prepend titles and subtitles to each section's text, to help GPT understand the context
- If a section is long (say, > 1,600 tokens), we'll recursively split it into smaller sections, trying to split along semantic boundaries like paragraphs


```python
# define functions to split Wikipedia pages into sections

SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
]


def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode,
    parent_titles: list[str],
    sections_to_ignore: set[str],
) -> list[tuple[list[str], str]]:
    """
    From a Wikipedia section, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    if title.strip("=" + " ") in sections_to_ignore:
        # ^wiki headings are wrapped like "== Heading =="
        return []
    titles = parent_titles + [title]
    full_text = str(section)
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(all_subsections_from_section(subsection, titles, sections_to_ignore))
        return results


def all_subsections_from_title(
    title: str,
    sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,
    site_name: str = WIKI_SITE,
) -> list[tuple[list[str], str]]:
    """From a Wikipedia page title, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """
    site = mwclient.Site(site_name)
    page = site.pages[title]
    text = page.text()
    parsed_text = mwparserfromhell.parse(text)
    headings = [str(h) for h in parsed_text.filter_headings()]
    if headings:
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        summary_text = str(parsed_text)
    results = [([title], summary_text)]
    for subsection in parsed_text.get_sections(levels=[2]):
        results.extend(all_subsections_from_section(subsection, [title], sections_to_ignore))
    return results

```


```python
# split pages into sections
# may take ~1 minute per 100 articles
wikipedia_sections = []
for title in titles:
    wikipedia_sections.extend(all_subsections_from_title(title))
print(f"Found {len(wikipedia_sections)} sections in {len(titles)} pages.")

```

    Found 5730 sections in 731 pages.



```python
# clean text
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    """
    Return a cleaned up section with:
        - <ref>xyz</ref> patterns removed
        - leading/trailing whitespace removed
    """
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return (titles, text)


wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]

# filter out short/blank sections
def keep_section(section: tuple[list[str], str]) -> bool:
    """Return True if the section should be kept, False otherwise."""
    titles, text = section
    if len(text) < 16:
        return False
    else:
        return True


original_num_sections = len(wikipedia_sections)
wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
print(f"Filtered out {original_num_sections-len(wikipedia_sections)} sections, leaving {len(wikipedia_sections)} sections.")

```

    Filtered out 530 sections, leaving 5200 sections.



```python
# print example data
for ws in wikipedia_sections[:5]:
    print(ws[0])
    display(ws[1][:77] + "...")
    print()

```

    ['Lviv bid for the 2022 Winter Olympics']



    '{{Olympic bid|2022|Winter|\n| Paralympics = yes\n| logo = Lviv 2022 Winter Olym...'


    
    ['Lviv bid for the 2022 Winter Olympics', '==History==']



    '[[Image:Lwów - Rynek 01.JPG|thumb|right|200px|View of Rynok Square in Lviv]]\n...'


    
    ['Lviv bid for the 2022 Winter Olympics', '==Venues==']



    '{{Location map+\n|Ukraine\n|border =\n|caption = Venue areas\n|float = left\n|widt...'


    
    ['Lviv bid for the 2022 Winter Olympics', '==Venues==', '===City zone===']



    'The main Olympic Park would be centered around the [[Arena Lviv]], hosting th...'


    
    ['Lviv bid for the 2022 Winter Olympics', '==Venues==', '===Mountain zone===', '====Venue cluster Tysovets-Panasivka====']



    'An existing military ski training facility in [[Tysovets, Skole Raion|Tysovet...'


    


Next, we'll recursively split long sections into smaller sections.

There's no perfect recipe for splitting text into sections.

Some tradeoffs include:
- Longer sections may be better for questions that require more context
- Longer sections may be worse for retrieval, as they may have more topics muddled together
- Shorter sections are better for reducing costs (which are proportional to the number of tokens)
- Shorter sections allow more sections to be retrieved, which may help with recall
- Overlapping sections may help prevent answers from being cut by section boundaries

Here, we'll use a simple approach and limit sections to 1,600 tokens each, recursively halving any sections that are too long. To avoid cutting in the middle of useful sentences, we'll split along paragraph boundaries when possible.


```python
GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

```


```python
# split sections into chunks
MAX_TOKENS = 1600
wikipedia_strings = []
for section in wikipedia_sections:
    wikipedia_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

print(f"{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings.")

```

    5200 Wikipedia sections split into 6059 strings.



```python
# print example data
print(wikipedia_strings[1])

```

    Lviv bid for the 2022 Winter Olympics
    
    ==History==
    
    [[Image:Lwów - Rynek 01.JPG|thumb|right|200px|View of Rynok Square in Lviv]]
    
    On 27 May 2010, [[President of Ukraine]] [[Viktor Yanukovych]] stated during a visit to [[Lviv]] that Ukraine "will start working on the official nomination of our country as the holder of the Winter Olympic Games in [[Carpathian Mountains|Carpathians]]".
    
    In September 2012, [[government of Ukraine]] approved a document about the technical-economic substantiation of the national project "Olympic Hope 2022". This was announced by Vladyslav Kaskiv, the head of Ukraine´s Derzhinvestproekt (State investment project). The organizers announced on their website venue plans featuring Lviv as the host city and location for the "ice sport" venues, [[Volovets]] (around {{convert|185|km|mi|abbr=on}} from Lviv) as venue for the [[Alpine skiing]] competitions and [[Tysovets, Skole Raion|Tysovets]] (around {{convert|130|km|mi|abbr=on}} from Lviv) as venue for all other "snow sport" competitions. By March 2013 no other preparations than the feasibility study had been approved.
    
    On 24 October 2013, session of the Lviv City Council adopted a resolution "About submission to the International Olympic Committee for nomination of city to participate in the procedure for determining the host city of Olympic and Paralympic Winter Games in 2022".
    
    On 5 November 2013, it was confirmed that Lviv was bidding to host the [[2022 Winter Olympics]]. Lviv would host the ice sport events, while the skiing events would be held in the [[Carpathian]] mountains. This was the first bid Ukraine had ever submitted for an Olympic Games.
    
    On 30 June 2014, the International Olympic Committee announced "Lviv will turn its attention to an Olympic bid for 2026, and not continue with its application for 2022. The decision comes as a result of the present political and economic circumstances in Ukraine."
    
    Ukraine's Deputy Prime Minister Oleksandr Vilkul said that the Winter Games "will be an impetus not just for promotion of sports and tourism in Ukraine, but a very important component in the economic development of Ukraine, the attraction of the investments, the creation of new jobs, opening Ukraine to the world, returning Ukrainians working abroad to their motherland."
    
    Lviv was one of the host cities of [[UEFA Euro 2012]].


## 3. Embed document chunks

Now that we've split our library into shorter self-contained strings, we can compute embeddings for each.

(For large embedding jobs, use a script like [api_request_parallel_processor.py](api_request_parallel_processor.py) to parallelize requests while throttling to stay under rate limits.)


```python
# calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

embeddings = []
for batch_start in range(0, len(wikipedia_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = wikipedia_strings[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": wikipedia_strings, "embedding": embeddings})

```

    Batch 0 to 999
    Batch 1000 to 1999
    Batch 2000 to 2999
    Batch 3000 to 3999
    Batch 4000 to 4999
    Batch 5000 to 5999
    Batch 6000 to 6999


## 4. Store document chunks and embeddings

Because this example only uses a few thousand strings, we'll store them in a CSV file.

(For larger datasets, use a vector database, which will be more performant.)


```python
# save document chunks and embeddings

SAVE_PATH = "data/winter_olympics_2022.csv"

df.to_csv(SAVE_PATH, index=False)

```
