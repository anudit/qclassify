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

There is no casing augmentation (0.2.1): the feature extractor lowercases every input before hashing
and computing shape features, so casing variants would be duplicates. Validation and test inputs keep their
authored casing, and are lowercased only inside the extractor like all other input.

Random splitting does not isolate shared surnames, related technologies or
similar phrasing. The metrics measure this synthetic distribution and can be
optimistic. Independent real HHQ queries, including casing and language slices,
are needed before relying on the model for routing.

`parity.json` stores 223 preprocessing/output probes for Python/TypeScript/WebGL
compatibility. They have no accuracy labels and are not used to train or select
a model. Their probability reference is the **packed-model ONNX**.

## Mined from Hackathon_People (added in 0.2.0)

`scripts/mine_people.py` + `scripts/merge_mined.py` add **14,029 examples** (8,271 `person_name`, 5,758
`search_query`) taken from the ~950k hackathon profiles and project titles behind Dumpyard
(local only: the 19,938-example corpus and `data/mined/` hold real people's names from scraped profiles, so they are **not committed** to this public repo; `examples.json`, `names.txt`, `queries.txt` and the split files here are still the 5,909 authored seed examples. Rebuild with the two scripts below).

- **Names:** real profile names (2-4 tokens, recurring first name, no org/topic words), and **one-word
  names**: first names that recur on >= 6 profiles ("Vitalik", "Subh"), excluding surnames, ordinary
  dictionary words and anything that is also a skill/city/company.
- **Queries:** skills, interests, job titles, companies, cities (one word and short phrases) and
  multi-word project titles. One-word queries come only from vocabulary, never from invented project names.
- **Held out:** `data/mined/probes_mined.json` (local; never trained on) plus `scripts/compare.py`, which scores
  any `classifier.onnx` on it and on `probes.json`.

These are real-world but weakly labelled (a profile "name" can be a handle or a team; a project title can
look like a name), so the mined slice has label noise and the overall test accuracy (~0.89) is lower than the
old synthetic corpus's, which was easier.

`artifacts/metrics.json` lists misclassified authored examples only; mined names are scrubbed from it.

## Authored topic phrases (added in 0.2.1)

`data/authored/topics.txt` holds ~575 hand-written short topic queries ("solar power", "ancient greece",
"grant proposals", "Wind Energy Basics") covering energy, science, health, food, home, work, culture, sport and
tech, including phrases built from words that are also first names or surnames. 0.2.0 scored "solar energy" 0.80
for person because short lowercase noun phrases looked like names. `scripts/merge_authored.py` folds the file
into `examples.json` and `queries.txt` (idempotent, skips held-out probes) and works on the committed corpus too.
`data/probes_topics.json` is a held-out set of 98 topic phrases never trained on; `scripts/compare.py` reports it
(0.776 for 0.2.0, 0.847 for 0.2.1).

`data/authored/famous.txt` (~320 notable people across science, computing, arts, politics, sport and business) was
added in 0.3.0 as `person_name`; `data/probes_famous.json` holds 28 held-out famous names (plus 6 name+request
queries). Adding hand-picked "famous name + request" queries made bare famous names worse (the model treated them as
nearly the same input) and was reverted. Training weighs hand-authored rows 3x over mined ones; see the main README.
