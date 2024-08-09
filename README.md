# Bouzyges

Bouzyges (pronounced boo-zee-jes) is a Python program to interactively generate a semantic graph of a medical term utilizing the SNOMED CT attribute-value pairs. The script can be interfaced with a LLM model to generate graphs in automated fashion. End result of the script is a set of SNOMED CT concepts, that serve as the closest possible strict supertypes that together fully capture the meaning of the input term.

# Intended use

In current form, Bouzyges serves as a proof-of-concept of a novel approach to automating ontology mapping and standardization. In the future, possible applications include:

- Mapping of medical terms to SNOMED CT concepts
- SNOMED CT authoring support
- Automated SNOMED CT quality assurance
- Automated creation of custom local Standard concepts in OMOP CDM.

# Installation

Bouzyges requires Python 3.12 or later. To install the script, clone the repository, initialize a virtual environment and install the required packages:

```bash
$ git clone https://github.com/OHDSI/Bouzyges.git
$ cd Bouzyges
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
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

Bouzyges relies on outputting LLM prompts and parsing their input; currently, two options are supported:

- Manual input: the user is prompted to input the desired LLM prompt and is expected to provide the input manually.
- Automated input: currently, only [Azure OpenAI](https://platform.openai.com/docs/libraries/azure-openai-libraries) API is supported. To use the API, you need to provide the endpoint and the API key either as an explicitly set environment variables or (preferred way) inside `.env` file in the root directory of the project. The file should contain definitions for the following environment variables:

```bash
SNOWSTORM_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_ENDPOINT=
```

# Usage

> [!WARNING]
> Bouzyges is currently in the early development stage and is not yet ready for production use. The script makes a lot of API calls and may consume a LOT of tokens. Profiling token usage and optimizing the script is currently in progress.

Currently, only exemplary usage inside the script is supported; batch loading interface is planned to be implemented very soon. To run the script, execute the following command:

```bash
$ python bouzyges.py
```

# License

The code is not yet licensed and is provided as-is. The code is provided for educational purposes only and is not intended for production use.

# Current work in progress

- [ ] OpenAI token consumption profiling
- [ ] Batch processing interface
- [ ] OpenAI API token consumption optimization
- [ ] Reproducible run instructions
- [ ] Licensing and release preparation
- [ ] SNOMED CT API optimization
- [ ] RAG support with SNOMED authoring documentation
- [x] Automated LLM interface
- [x] SNOMED CT API interface
- [x] SNOMED CT hierarchy traversal
