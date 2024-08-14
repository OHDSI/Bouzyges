# Bouzyges

Bouzyges (pronounced boo-zee-jes) is a Python program to interactively generate semantic graphs of medical terms utilizing the SNOMED CT attribute-value pairs. The script can be interfaced with a LLM model to generate graphs in automated fashion. End result of the script is a set of SNOMED CT concepts, that serve as the closest possible strict supertypes that together fully capture the meaning of the input term.

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

Current implementation of Bouzyges relies on Snowstorm REST API to interface with SNOMED CT. To use the API, you need to provide the endpoint and the API key either as environment variables or inside `.env` file in the root directory of the project (see below).

Snowstorm version 10 with SNOMED International (July 2024 release) was tested. We recommend using the Docker image provided by SNOMED International to run Snowstorm locally and loading the SNOMED RF2 release archive via Swagger UI.

### External links:

- [Snowstorm GitHub repository](https://github.com/IHTSDO/snowstorm)
- [Using Snowstorm with Docker](https://github.com/IHTSDO/snowstorm/blob/master/docs/using-docker.md)
- [SNOMED International release in RF2 format](https://www.nlm.nih.gov/healthit/snomedct/international.html) (hosted by NLM)

## LLM interface

Bouzyges relies on outputting LLM prompts and parsing their input; currently, three options are supported:

- Manual input: the user is prompted to input the desired LLM prompt and is expected to provide the input manually. This can be used to debug the script or test different LLMs interactively. To use this, set `PROMPTER_OPTION` constant to `"human"` in the body of the script. Better configuration interface
is coming soon.
- [OpenAI](https://platform.openai.com/docs/api-reference/introduction): to use this API, you will need to ensure that a valid `OPENAI_API_KEY` is set either as environment variable or (recommended) in `env` file (see below). To use this, set `PROMPTER_OPTION` to `"openai"`
- Azure: [Azure OpenAI](https://platform.openai.com/docs/libraries/azure-openai-libraries) API is also supported. To use this API, you will need to provide the API information either an by explicitly setting environment variables or (preferred way) inside `.env` file. The `PROMPTER_OPTION` should be set to `"azure"`.


### Implementing new interfaces
It is possible to implement additional API interfaces (e.g. to locally available models) by inheriting from `PromptFormat` class to generate prompts in the correct format in inheriting from `Prompter` to provide interface to send prompts to the LLM.

## `.env` file
To avoid accidental exposure of API keys, we strongly recommend using [an `.env` file](https://hexdocs.pm/dotenvy/dotenv-file-format.html) to manage environment variables. Bouzyges will try to automatically load the `.env` file in the working directory using the [python-dotenv](https://pypi.org/project/python-dotenv/) library.

Example content of the file:
```bash
# Snowstorm endpoint is always required
# This is example for default local/docker installation is given
export SNOWSTORM_ENDPOINT="https://localhost:8080/"

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
> Bouzyges is currently in the early development stage and is not yet ready for production use. The script makes a lot of API calls and may consume a LOT of tokens. Currently, processing one concept consumes tokens on magnitude of 150,000 (3 cents with gpt-4o-mini).

Currently, only exemplary usage inside the script is supported; batch loading interface is planned to be implemented very soon. To run the script, execute the following command:

```bash
$ python bouzyges.py
```

# License

The code is not yet licensed and is provided as-is. The code is provided for educational purposes only and is not intended for production use.

# Current work in progress

- [ ] Batch processing interface
- [ ] Reproducible run instructions
- [ ] Licensing and release preparation
- [ ] RAG support with SNOMED authoring documentation
- [x] SNOMED CT API optimization
- [x] OpenAI token consumption profiling
- [x] OpenAI API token consumption optimization
- [x] Automated LLM interface
- [x] SNOMED CT API interface
- [x] SNOMED CT hierarchy traversal
