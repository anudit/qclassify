# Dataset

`examples.json` contains **5,909 explicitly authored examples**, each with:

```json
{"input": "Aarav Sharma", "output": "person_name"}
```

- **2,915 `person_name`** examples: names across multiple naming traditions,
  initials, apostrophes, hyphens, accents, non-Latin scripts, mononyms, lowercase
  names, and public technology/research figures relevant to HHQ.
- **2,994 `search_query`** examples: general topics, questions, developer products,
  research terms, technologies, project/people discovery, brand names, and
  searches containing a person's name.

These examples were written explicitly by the coding assistant. They are
synthetic seed data, not human-annotated user traffic, scraped profiles, Faker
output, or an expansion of name/query templates. Fictional names can coincidentally
match real people. `names.txt` and `queries.txt` preserve the authored strings;
the JSON assigns each string its label. JSON and text sources should be edited
together; tests check that they agree. No data generator is required to train.

HHQ context was read from `dumpyard/app/hhq/search.tsx` and `projects.ts`: the
interface distinguishes person lookups from topics and returns projects,
opportunities, articles, ideas, and people. HHQ's private runtime data was not
copied. Its code was used only to understand the task.

## Label policy

A bare personal name is `person_name`, even if searching for that name is also
a search in ordinary language. `Ada Lovelace` is a name; `Ada Lovelace biography`
is a `search_query`. A product such as `Hugging Face` is a query. A topic remains
a query in title case. Empty strings are rejected, not assigned a training label.

Bare ambiguous words such as `Prince` are labeled under a name-plausibility
convention. Text alone cannot infer the user's actual intent. Handles, URLs,
company names resembling names, incomplete names and prefixes need validation
against actual HHQ traffic; these are not a comprehensive identity recognizer.

`probes.json` holds held-out lowercase/title-case names and queries kept out of
training and out of the corpus (the loader rejects duplicates against them in curation).

## Splitting and augmentation

`data/splits.json` records the deterministic stratified 80/10/10 split: **4,727
training / 590 validation / 592 test** inputs. Normalized duplicate inputs are
rejected across the entire authored corpus. Sorting before shuffling means
reordering the JSON does not change partition membership.

`data/augmented_train.json` stores the training-only candidate pool: original,
lowercase, uppercase, and title-case versions (**18,908 rows**). These generated
variants are separate from the explicitly authored seed examples. Each epoch
samples 55% original / 35% lower / 5% upper / 5% title variants; validation and
test inputs retain their authored casing. Seeds and augmentation settings are
recorded in the model report. The augmentation pool deliberately contains
repeated inputs; it is not a second independently authored dataset.

Random splitting does not isolate shared surnames, related technologies or
similar phrasing. The metrics measure this synthetic distribution and can be
optimistic. Independent real HHQ queries, including casing and language slices,
are needed before relying on the model for routing.

`parity.json` stores 223 preprocessing/output probes for Python/TypeScript/WebGL
compatibility. They have no accuracy labels and are not used to train or select
a model. Their probability reference is the **packed-model ONNX**.
