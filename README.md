# Bouzyges

Bouzyges (pronounced boo-zee-jes) is a Python program to interactively generate semantic graphs of medical terms utilizing the SNOMED CT attribute-value pairs. The script can be interfaced with a LLM model to generate graphs in automated fashion. End result of the script is a set of SNOMED CT concepts, that serve as the closest possible strict supertypes that together fully capture the meaning of the input term.

# Citing Bouzyges
Authors of Bouzyges kindly ask you to cite the following conference publication if you use Bouzyges in your research or publish a derived work:
[https://www.ohdsi.org/2024showcase-32/](https://www.ohdsi.org/2024showcase-32/)

We include a BibTeX and a modern [Hayagriva](https://github.com/typst/hayagriva/blob/main/docs/file-format.md) citation snippets for your convenience:

## BibTeX
```bibtex
@inproceedings{Bouzyges,
    title = "Automating data standardization through ad hoc SNOMED modeling with LLM: proof of concept",
    author = "Eduard Korchmar and Vojtech Huser and Christian Reich and Alexander Davydov",
    howpublished = "\url{https://www.ohdsi.org/2024showcase-32/}",
    organization = "OHDSI",
    type = "Collaboration Showcase",
    booktitle = "OHDSI 2024 Global Symposium",
    conference = "OHDSI 2024 Global Symposium",
    year = "2024",
    month = "October",
    day = "20",
    address = "New Brunswick, NJ, USA"
}
```

## Hayagriva
```yaml
bouzyges:
    type: article
    title: "Automating data standardization through ad hoc SNOMED modeling with LLM: proof of concept"
    author:
        - Eduard Korchmar
        - Vojtech Huser
        - Christian Reich
        - Alexander Davydov
    date: 2024-10-20
    url: https://www.ohdsi.org/2024showcase-32/
    parent:
        type: conference
        title: OHDSI 2024 Global Symposium
        organization: OHDSI
        address: New Brunswick, NJ, USA
```

# Intended use

In current form, Bouzyges serves as a proof-of-concept of a novel approach to automating ontology mapping and standardization. In the future, possible applications include:

- Mapping of medical terms to SNOMED CT concepts
- SNOMED CT authoring support
- Automated SNOMED CT quality assurance
- Automated creation of custom local Standard concepts in OMOP CDM.

# Installation

Bouzyges requires Python 3.12 or later. To install the script, clone the repository, initialize a virtual environment and install the required packages:

```bash
git clone https://github.com/OHDSI/Bouzyges.git
cd Bouzyges
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Prerequisites

## SNOMED CT

Current implementation of Bouzyges relies on Snowstorm REST API to interface with SNOMED CT. To use the API, you need to provide the endpoint for the Snowstorm service in `default_config.json` file or in GUI configuraiton.

Snowstorm version 10 with SNOMED International (July 2024 release) was tested. We recommend using the Docker image provided by SNOMED International to run Snowstorm locally and loading the SNOMED RF2 release archive via the Swagger UI.

### External links:

- [Snowstorm GitHub repository](https://github.com/IHTSDO/snowstorm)
- [Using Snowstorm with Docker](https://github.com/IHTSDO/snowstorm/blob/master/docs/using-docker.md)
- [SNOMED International release in RF2 format](https://www.nlm.nih.gov/healthit/snomedct/international.html) (hosted by NLM)

## LLM interface

Bouzyges relies on outputting LLM prompts and parsing their input; currently, three options are supported:

- Manual input: the user is prompted to input the desired LLM prompt and is expected to provide the input manually. This can be used to debug the script or test different LLMs interactively.
- [OpenAI](https://platform.openai.com/docs/api-reference/introduction): to use this API, you will need to ensure that a valid `OPENAI_API_KEY` is set either as environment variable or (recommended) in `env` file (see below). You can also set environment variables in Bouzyges GUI.
- Azure: [Azure OpenAI](https://platform.openai.com/docs/libraries/azure-openai-libraries) API is also supported. To use this API, you will need to provide the API information either an by explicitly setting environment variables or (preferred way) inside `.env` file.


### Implementing new interfaces
It is possible to implement additional API interfaces (e.g. to locally available models) by inheriting from `PromptFormat` class to generate prompts in the correct format and inheriting from `Prompter` to provide interface to send prompts to the LLM.

## `.env` file
To avoid accidental exposure of API keys, we strongly recommend using [an `.env` file](https://hexdocs.pm/dotenvy/dotenv-file-format.html) to manage environment variables. Bouzyges will try to automatically load the `.env` file in the working directory using the [python-dotenv](https://pypi.org/project/python-dotenv/) library. You can also manully paste API keys in Bouzyges interface in `Edit > Override environment variables` menu.

Example content of the `.env` file:
```bash
# OpenAI requirements
# Project API key created at https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-abc...def"

# Azure OpenAI interface requirements
# Attainable at your organization's infrastructure team
export AZURE_OPENAI_API_KEY="123abcd...789"
export AZURE_OPENAI_API_VERSION="2024-06-01"  # Most recent version
export AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/
```

## Caching of results
Bouzyges will cache all calls to LLM APIs in an SQLite database `prompt_cache.db`. Prompts to the same model with the same API options will be reused across runs. Database file can be read and analyzed by any tool supporting sqlite3 APIs. Schema DDL is stored in `init_prompt_cache.sql` file.


# Usage

> [!WARNING]
> Bouzyges makes a lot of API calls and may consume a LOT of tokens. Currently, processing one concept consumes tokens on magnitude of 150,000 (3 cents with gpt-4o-mini). Performant models like GPT-4o or GPT-1o will be more expensive by orders of magnitude.

To run the script, execute the following command:

```bash
$ python bouzyges.py
```
This will start the GUI interface.

Bouzyges can be configured either interactively in GUI or by editing `default_config.json` file.

## `default_config.json` file

Note that sensitive data should not be stored in the configuration file. API keys and Azure endpoints should be stored in `.env` file.

The configuration file is divided into several blocks:

### API block
API options contain fields to configure connections to OpenAI API. There is also an option to set the number of prompt repeats (each query is repeated several times to get "best of N" answer) and the number of concurrent threads to allow for parallel processing.

### Log block
Contatains options to enable logging to a file (default to true) and logging level (default to DEBUG -- 10).

### Profile block
Contains options to enable profiling of the script with default cProfile module. Should be considered deprecated, as it is not informative about multithreaded performance.

### Read block
Contains options to enable reading from a file, and the file name to read from. Contains fields that will be passed to `pandas.read_csv` function.

### Write block
Contains options to enable writing to a file, and the file name to write to. Contains fields that will be passed to `DataFrame.to_csv` method. If JSON format is chosen for output, options will be ignored.

### Format field
Can be either "JSON", "CSV" or "SCG". "JSON" option is recommended.


# License

By necessity of license requirements of QT graphical library, Bouzyges is licensed under GNU GPL v3.0. Please refer to the `LICENSE` file for more information. GNU GPL v3.0 is a strong copyleft license, any derivative works must also be licensed under GNU GPL v3.0, which may not be suitable for commercial use-cases.

We intend to release an embeddable version of Bouzyges under a more permissive and derivation-friendly license in the near future.

# Current work in progress

- [ ] Refactoring into submodules
- [ ] Headless library mode re-release under a softer license
- [ ] Switch to OpenAI batch processing API
- [ ] Benchmarking framework and iteration on prompts
- [ ] RAG support with SNOMED authoring documentation
- [x] Batch processing interface
- [x] Reproducible run instructions
- [x] Licensing and release preparation
- [x] SNOMED CT API optimization
- [x] OpenAI token consumption profiling
- [x] OpenAI API token consumption optimization
- [x] Automated LLM interface
- [x] SNOMED CT API interface
- [x] SNOMED CT hierarchy traversal
