from __future__ import annotations

import asyncio
import cProfile
import csv
import datetime
import json
import logging
import os
import pstats
import re
import sys
import unittest
from snomed import SnowstormAPI
from utils.constants import (
    AVAILABLE_PROMPTERS,
    CSV_SEPARATORS,
    QUOTING_POLICY,
    DEFAULT_MODEL,
)
from parameters import EnvironmentParameters, IOParameters, PARAMS

from utils.logger import LOGGER, FORMATTER
from file_io import FileWriter, FileReader

from typing import (
    Callable,
    Iterable,
    Literal,
    Self,
)
from utils.exceptions import (
    ProfileMark,
    BouzygesError,
)

import httpx
import pandas as pd
import webbrowser
from frozendict import frozendict
import qasync
from PyQt6 import QtCore, QtGui, QtWidgets
from prompt import (
    OpenAIPromptFormat,
    VerbosePromptFormat,
)
from utils.constants import DEFAULT_REPEAT_PROMPTS
from utils.types import (
    Url,
    SCTID,
    EscapeHatch,
    SCTDescription,
)

from prompter import (
    Prompter,
    OpenAIAzurePrompter,
    OpenAIPrompter,
    HumanPrompter,
)
from models import (
    Concept,
    AttributeConstraints,
    SemanticPortrait,
    WrappedResult,
)

# Optional imports
## dotenv
try:
    import dotenv
except ImportError:
    logging.info(
        "python-dotenv package not installed. "
        "Relying on explicit environment variables"
    )
    dotenv = None

## Parameters
# Load environment variables for API access
if dotenv is not None:
    LOGGER.info("Loading environment variables from .env")
    if os.path.exists(".env"):
        dotenv.load_dotenv()
    else:
        LOGGER.warning("No .env file found")


class IOParametersWidget(QtWidgets.QWidget):
    def __init__(self, par: IOParameters, name: str, *args, **kwargs) -> None:
        # Add pydantic fields
        super().__init__(*args, **kwargs)

        self.logger = LOGGER.getChild(name)
        self.parameters: IOParameters = par
        self._populate_layout()
        self.set_values()

    def _populate_layout(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        self.sep_cb = QtWidgets.QComboBox()
        self.sep_cb.addItems(map(lambda s: "Separator: " + s, CSV_SEPARATORS))
        self.sep_cb.currentIndexChanged.connect(self.separator_changed)
        layout.addWidget(self.sep_cb)
        self.quoting_cb = QtWidgets.QComboBox()
        self.quoting_cb.addItems(QUOTING_POLICY.values())
        self.quoting_cb.currentIndexChanged.connect(self.quoting_policy_changed)
        layout.addWidget(self.quoting_cb)
        qchar_label = QtWidgets.QLabel("Quote character:")
        layout.addWidget(qchar_label)
        self.qc_edit = QtWidgets.QLineEdit()
        self.qc_edit.setPlaceholderText('"')
        self.qc_edit.setMaximumWidth(30)
        self.qc_edit.textChanged.connect(self.quote_char_changed)
        layout.addWidget(self.qc_edit)
        spacer = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        layout.addItem(spacer)
        self.setLayout(layout)

    def separator_changed(self, index) -> None:
        self.parameters.sep = list(CSV_SEPARATORS)[index]
        self.logger.debug(f"Separator changed to: {self.parameters.sep}")

    def quoting_policy_changed(self, index) -> None:
        self.parameters.quoting = list(QUOTING_POLICY)[index]
        self.logger.debug(
            f"Quoting policy changed to: {self.parameters.quoting}:"
            f"{QUOTING_POLICY[self.parameters.quoting]}"
        )
        self.qc_edit.setEnabled(self.parameters.quoting != csv.QUOTE_NONE)

    def quote_char_changed(self, text) -> None:
        self.parameters.quotechar = text
        self.logger.debug(f"Quote character changed to: {repr(text)}")

    def update_file(
        self, file: str, update: Callable[[str], None] | None = None
    ) -> None:
        self.parameters.file = file
        if update:
            update(file)
        self.logger.debug(f"Input file set to: {file}")

    def set_values(self) -> None:
        idx = list(CSV_SEPARATORS).index(self.parameters.sep)
        self.sep_cb.setCurrentIndex(idx)
        self.quoting_cb.setCurrentIndex(
            list(QUOTING_POLICY).index(self.parameters.quoting)
        )
        self.qc_edit.setText(self.parameters.quotechar)
        self.qc_edit.setEnabled(self.parameters.quoting != csv.QUOTE_NONE)


LOGGER.info(f"Parameters loaded: {json.dumps(PARAMS.model_dump(), indent=2)}")
LOGGER.setLevel(PARAMS.log.logging_level)


## Hacked Httpx client
# HACK: Somehow, for whatever reason, the connection pool of Httpx client
# is constantly filling up with unusable connections. This is a hack to
# flush the pool on a timeout and continue.
class HackedAsyncClient(httpx.AsyncClient):
    """\
Hacked Httpx client to flush connection pool on timeout.
"""

    async def send(self, *args, **kwargs):
        try:
            return await super().send(*args, **kwargs)
        except httpx.HTTPError as e:
            transport: httpx.AsyncHTTPTransport = self._transport  # type: ignore
            pool = transport._pool
            conns = pool.connections
            bad_connections = {
                "closed": [],
                "expired": [],
                "idle": [],
            }
            for conn in conns:
                if conn.is_closed:
                    bad_connections["closed"].append(conn)
                elif conn.has_expired:
                    bad_connections["expired"].append(conn)
                elif conn.is_idle:
                    bad_connections["idle"].append(conn)
            LOGGER.error(
                f"Failed to connect: {type(e)}. Flushing connections "
                f"from AsyncClient",
                exc_info=e,
            )
            for reason, conns in bad_connections.items():
                if not conns:
                    continue
                LOGGER.error(f"{len(conns)} connections to close: {reason}")
                await pool._close_connections(conns)
                for connection in conns:
                    pool._connections.remove(connection)
            raise
        except Exception as e:
            LOGGER.error(f"Failed to connect: {type(e)}", exc_info=e)
            raise


class TestFileReader(unittest.TestCase):
    def setUp(self) -> None:
        self.row = pd.Series(
            {
                "vocab": "ICD-42",
                "code": 123456,
                "term": "Test term",
                "foo": "Test context",
                "bar": "Another context",
            }
        )
        return super().setUp()

    def test_parse(self):
        portrait = FileReader._portrait_from_row(self.row)
        self.assertEqual(portrait.source_term, "Test term")
        self.assertIsNot(portrait.context, None)
        self.assertEqual(
            set(portrait.context),  # type: ignore
            {"Test context", "Another context"},
        )
        self.assertEqual(
            portrait.metadata, frozendict(vocab="ICD-42", code="123456")
        )


# Main logic host
class Bouzyges:
    """\
Main logic host for the Bouzyges system.
"""

    def __init__(
        self,
        snowstorm: SnowstormAPI,
        prompter: Prompter,
        portraits: Iterable[SemanticPortrait],
    ):
        self.snowstorm = snowstorm
        self.prompter = prompter
        self.portraits = {p.source_term: p for p in portraits}

        self.logger = LOGGER.getChild(self.__class__.__name__)

        self.results: list[WrappedResult] = []

    @staticmethod
    async def read_file(
        logger: logging.Logger,
        http_client: httpx.AsyncClient,
        prep_dict: dict,
        ready_callback: Callable,
    ) -> None:
        """\
Read the input file and parse it into a list of SemanticPortrait objects.
"""
        _ = http_client
        # Read file for portraits
        if not PARAMS.read.file:
            raise BouzygesError("No input file specified!")
        logger.info("Reading input file...")
        reader = FileReader(PARAMS.read)
        reader.read()
        if reader.content is None:
            raise BouzygesError("Could not read the input file")
        portraits = reader.parse()
        logger.info(f"Read {len(portraits)} portraits")
        prep_dict["portraits"] = portraits
        logger.info("FILE READ")
        ready_callback()

    @staticmethod
    async def get_snowstorm(
        logger: logging.Logger,
        http_client: httpx.AsyncClient,
        prep_dict: dict,
        ready_callback: Callable,
    ) -> None:
        """\
Initialize the SnowstormAPI object.
"""
        logger.info("Initializing Snowstorm API...")
        try:
            snowstorm = await SnowstormAPI.init(
                PARAMS.api.snowstorm_url, http_client
            )
        except Exception as e:
            logger.error("Could not connect to Snowstorm API", exc_info=e)
            raise
        prep_dict["snowstorm"] = snowstorm
        logger.info("SNOWSTORM API INITIALIZED")
        ready_callback()

    @staticmethod
    async def get_prompter(
        logger: logging.Logger,
        http_client: httpx.AsyncClient,
        prep_dict: dict,
        ready_callback: Callable,
    ):
        logger.info("Initializing prompter...")
        repeat_prompts = (
            DEFAULT_REPEAT_PROMPTS
            if PARAMS.api.repeat_prompts is None
            else PARAMS.api.repeat_prompts
        )
        prompter: Prompter
        match PARAMS.api.prompter:
            case "openai":
                prompter = OpenAIPrompter(
                    prompt_format=OpenAIPromptFormat(),
                    api_parameters=PARAMS.api,
                    http_client=http_client,
                    repeat_prompts=repeat_prompts,
                    model=PARAMS.api.llm_model_id,
                )

            case "azure":
                prompter = OpenAIAzurePrompter(
                    prompt_format=OpenAIPromptFormat(),
                    api_parameters=PARAMS.api,
                    http_client=http_client,
                    repeat_prompts=repeat_prompts,
                    api_key=PARAMS.env.AZURE_API_KEY,
                    azure_endpoint=PARAMS.env.AZURE_API_ENDPOINT,
                    model=PARAMS.api.llm_model_id,
                )
            case "human":
                prompter = HumanPrompter(
                    api_parameters=PARAMS.api,
                    prompt_function=input,
                    prompt_format=VerbosePromptFormat(),
                )

            case _:
                raise ValueError("Invalid prompter option")
        prep_dict["prompter"] = prompter
        logger.info("PROMPTER INITIALIZED")
        ready_callback()

    @classmethod
    async def prepare(cls, progress_callback) -> Self:
        logger = LOGGER.getChild(cls.__name__)
        progress_callback(0, 3)

        prep_dict = {}

        def report_completion():
            progress_callback(len(prep_dict), 3)

        limits = httpx.Limits(
            max_connections=20, max_keepalive_connections=0, keepalive_expiry=0
        )
        timeout = httpx.Timeout(60.0)
        http_client = HackedAsyncClient(limits=limits, timeout=timeout)
        futures = []
        for task in (cls.get_snowstorm, cls.get_prompter, cls.read_file):
            futures.append(
                task(logger, http_client, prep_dict, report_completion)
            )

        await asyncio.gather(*futures)
        await asyncio.sleep(0.1)  # Let the progress bar catch up

        return cls(**prep_dict)

    async def _run(self, progress_callback) -> bool:
        """Main routine"""
        start_time = datetime.datetime.now()
        self.logger.info(f"Started at: {start_time}")

        workers = [
            _BouzygesWorker(i, portrait, self)
            for i, portrait in enumerate(self.portraits.values())
        ]

        done: list[_BouzygesWorker] = []

        def report_progress(worker: _BouzygesWorker):
            done.append(worker)
            progress_callback(len(done), len(workers))

        async with asyncio.Semaphore(PARAMS.api.max_concurrent_workers):
            try:
                self.results = await asyncio.gather(
                    *map(lambda w: w.run(report_progress), workers)
                )
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                return False

        self.logger.info("Routine finished")
        self.logger.info(
            f"Time taken (s): "
            f"{(datetime.datetime.now() - start_time).total_seconds()}"
        )

        self.logger.info("Closing Snowstorm API connection")
        await self.snowstorm.async_client.aclose()

        return True

    async def run(self, progress_callback) -> bool:
        """\
Run the Bouzyges system.
"""
        if PARAMS.prof:
            with cProfile.Profile() as prof:
                try:
                    return await self._run(progress_callback)
                except ProfileMark:
                    return True
                finally:
                    stats = pstats.Stats(prof)
                    stats.sort_stats(pstats.SortKey.TIME)
                    stats.dump_stats("stats.prof")
        else:
            return await self._run(progress_callback)


class _BouzygesWorker:
    """\
Worker thread for the Bouzyges system, performing the main logic. on a single
source term.
"""

    def __init__(
        self, idx: int, portrait: SemanticPortrait, bouzyges: Bouzyges
    ):
        self.source_term = portrait.source_term
        self.portrait = portrait

        term_abbrev = "".join(
            map(
                lambda w: re.sub(r"\W", "", w)[0].upper(),
                self.source_term.split(),
            )
        )
        self.logger = bouzyges.logger.getChild(f"Worker {idx}:{term_abbrev}")
        self.snowstorm = bouzyges.snowstorm
        self.prompter = bouzyges.prompter
        self.__mrcm_entries = bouzyges.snowstorm.mrcm_entries

        self.logger.info(f"Worker for '{self.source_term}' is ready")
        self.writer = FileWriter(
            os.path.join(PARAMS.out_dir, PARAMS.write.file),
            write_parameters=PARAMS.write,
            format=PARAMS.format,
            append=True,
            logger=self.logger,
        )

    async def run(self, report_progress) -> WrappedResult:
        """\
Run the worker thread.
"""
        start_time = datetime.datetime.now()
        self.logger.info(f"Started at: {start_time}")
        try:
            return await self._run(report_progress)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=e)
            raise
        finally:
            self.logger.info(
                f"Time taken (s): "
                f"{(datetime.datetime.now() - start_time).total_seconds()}"
            )
            self.prompter.report_usage()

    async def _run(self, report_progress) -> WrappedResult:
        await self.initialize_supertypes()
        await self.populate_attribute_candidates()
        await self.populate_unchecked_attributes()
        await self.update_existing_attr_values()

        changes_made = updated = True
        while changes_made:
            cycles = 0
            while updated:
                updated = await self.update_anchors()
                await asyncio.sleep(0.05)
                cycles += updated
            changes_made = bool(cycles)

        await self.snowstorm.remove_redundant_ancestors(self.portrait)

        # Log resulting supertypes
        attr_message = []
        attr_message += [f"'{self.source_term}' Attributes:"]
        for attribute, value in self.portrait.attributes.items():
            attr_message.append(f" - {attribute}={value}")
        self.logger.info("\n".join(attr_message))

        supr_message = []
        supr_message += [f"'{self.source_term}' Supertypes:"]
        anchors: dict[SCTID, SCTDescription] = {}

        async def get_anchor_info(anchor):
            concept = await self.snowstorm.get_concept(anchor)
            supr_message.append(f" - {concept.sctid} {concept.pt}")
            anchors[concept.sctid] = concept.pt

        await asyncio.gather(
            *map(get_anchor_info, self.portrait.ancestor_anchors)
        )
        await asyncio.sleep(0.05)
        self.logger.info("\n".join(supr_message))

        result = WrappedResult(self.portrait, anchors)
        self.logger.info("Worker finished, writing result")
        await self.writer.write_chosen([result], self.snowstorm)
        report_progress(self)
        return result

    async def initialize_supertypes(self):
        """\
Initialize supertypes for all terms to start building portraits.
"""
        if self.portrait.ancestor_anchors:
            raise BouzygesError(
                "Should not happen: ancestor anchors are set, "
                "and yet initialize_supertypes is called"
            )

        supertypes_decode = {
            entry.term: entry.sctid for entry in self.__mrcm_entries
        }
        supertype_term = await self.prompter.prompt_supertype(
            self.source_term,
            supertypes_decode,
            False,
            "; ".join(self.portrait.context) if self.portrait.context else None,
        )
        match supertype_term:
            case SCTDescription(answer_term):
                supertype = supertypes_decode[answer_term]
                self.logger.info(
                    f"Assuming {self.source_term} is {answer_term}"
                )
                self.portrait.ancestor_anchors.add(supertype)
            case EscapeHatch.WORD:
                raise BouzygesError(
                    "Should not happen: null-like response from prompter; "
                    "did the Prompter inject the escape hatch?"
                )

    async def populate_attribute_candidates(self) -> None:
        attributes: dict[
            SCTID, AttributeConstraints
        ] = await self.snowstorm.get_attribute_suggestions(
            self.portrait.ancestor_anchors
        )

        # Remove previously rejected attributes
        for attribute in self.portrait.rejected_attributes:
            attributes.pop(attribute, None)

        possible_message = []
        possible_message.append("Possible attributes for: " + self.source_term)
        for sctid, attribute in attributes.items():
            possible_message.append(f" - {sctid} {attribute.pt}")
        self.logger.debug("\n".join(possible_message))

        # Confirm the attributes
        for attribute in attributes.values():
            accept: bool = await self.prompter.prompt_attr_presence(
                self.source_term,
                attribute.pt,
                "; ".join(self.portrait.context)
                if self.portrait.context
                else None,
            )
            self.logger.info(
                f"{attribute.sctid} {attribute.pt}: " + "Present"
                if accept
                else "Not present"
            )

            if accept:
                self.portrait.unchecked_attributes.add(attribute.sctid)

        # Remember the constraints
        for sctid, attribute in attributes.items():
            if sctid not in self.portrait.unchecked_attributes:
                continue
            self.portrait.relevant_constraints[sctid] = attribute

    async def populate_unchecked_attributes(self) -> None:
        rejected = set()
        for attribute in self.portrait.unchecked_attributes:
            self.logger.debug(f"Attribute: {attribute}")
            # Get possible attribute values
            values_options: dict[SCTID, SCTDescription] = (
                self.snowstorm.get_attribute_values(self.portrait, attribute)
            )
            self.logger.debug(f"Values: {values_options}")

            if not values_options:
                # No valid values for this attribute and parent combination
                rejected.add(attribute)
                continue
            else:
                possible_message = []
                possible_message.append(
                    f"Possible values for {attribute} in {self.source_term}"
                )
                for value in values_options:
                    possible_message.append(
                        f" - {value} {values_options[value]}"
                    )
                self.logger.debug("\n".join(possible_message))

            # Prompt for the value
            value_term: (
                SCTDescription | EscapeHatch
            ) = await self.prompter.prompt_attr_value(
                self.source_term,
                self.portrait.relevant_constraints[attribute].pt,
                values_options.values(),
                "; ".join(self.portrait.context)
                if self.portrait.context
                else None,
                allow_escape=True,
            )

            match value_term:
                case SCTDescription(answer_term):
                    sctid = next(
                        SCTID(sctid)
                        for sctid, term in values_options.items()
                        if term == answer_term
                    )
                    self.portrait.attributes[attribute] = sctid
                case EscapeHatch.WORD:
                    # Choosing no attribute on initial prompt
                    # means rejection
                    rejected.add(attribute)

        self.portrait.rejected_attributes |= rejected
        # All are seen by now
        self.portrait.unchecked_attributes = set()

    async def update_existing_attr_values(self) -> None:
        """\
Update existing attribute values with the most precise descendant for all terms.
"""
        new_attributes = {}
        for attribute, value in self.portrait.attributes.items():
            new_attributes[attribute] = value
            while True:
                # Get children of the current value
                children = await self.snowstorm.get_concept_children(
                    new_attributes[attribute]
                )
                if not children:
                    # Leaf node
                    break

                descriptions = {v: k for k, v in children.items()}

                # Prompt for the most precise value
                value_term: (
                    SCTDescription | EscapeHatch
                ) = await self.prompter.prompt_attr_value(
                    self.source_term,
                    attribute=self.portrait.relevant_constraints[attribute].pt,
                    options=descriptions,
                    term_context="; ".join(self.portrait.context)
                    if self.portrait.context
                    else None,
                    allow_escape=True,
                )

                if isinstance(value_term, EscapeHatch):
                    # None of the children are correct
                    break

                new_attributes[attribute] = descriptions[value_term]

        self.portrait.attributes.update(new_attributes)

    async def update_anchors(self) -> bool:
        i = 0
        ancestors_changed = True
        while ancestors_changed:
            ancestors_changed = await self.__update_anchor()
            i += ancestors_changed
        self.logger.info(
            f"Updated {self.source_term} anchors in {i} iterations."
        )
        return i > 1

    async def __update_anchor(self) -> bool:
        """\
Update the ancestor anchors for one term to more precise children. Performs a
single iteration. Return True if the parent anchors have changed, False
otherwise.
"""
        new_anchors = set()

        # Gather all children of currently known descendants:
        all_children: set[SCTID] = set()
        for anchor in self.portrait.ancestor_anchors:
            # Get all immediate descendants
            children_set: set[SCTID] = set(
                await self.snowstorm.get_concept_children(anchor)
            )
            all_children |= children_set

        self.logger.debug(f"Filtering {len(all_children)} children")
        # Remove verbatim known ancestors
        all_children -= self.portrait.ancestor_anchors

        # Filter previously rejected ancestors including meta-ancestors
        remaining = await self.snowstorm.filter_bad_descendants(
            children=all_children,
            bad_parents=self.portrait.rejected_supertypes,
        )

        # Save the rejected children as rejected supertypes
        self.portrait.rejected_supertypes.update(all_children - remaining)

        good_children: dict[SCTID, Concept] = await self.snowstorm.get_concepts(
            remaining
        )

        self.logger.debug(f"Filtered to {len(good_children)}")

        # Iterate over descendants and ask LLM/Snowstorm if to include them
        # to the new anchors
        for child in good_children.values():
            is_inferrable_supertype: bool = (
                await self.snowstorm.check_inferred_subsumption(
                    child, self.portrait
                )
            )

            if not is_inferrable_supertype:
                self.logger.debug(
                    f"{child.sctid} {child.pt} can not be inferred as a "
                    f"supertype of {self.source_term}"
                )
                self.portrait.rejected_supertypes.add(child.sctid)
                continue

            # Primitive concepts must be confirmed by the LLM
            primitive = not child.defined
            if primitive:
                source_term_context = (
                    "; ".join(self.portrait.context)
                    if self.portrait.context
                    else None
                )

                if not await self.prompter.prompt_subsumption(
                    self.portrait.source_term,
                    prospective_supertype=child.pt,
                    term_context=source_term_context,
                ):
                    self.logger.debug(
                        f"{child.sctid} {child.pt} is not a supertype of "
                        f"{self.source_term} according to the agent"
                    )
                    self.portrait.rejected_supertypes.add(child.sctid)
                    continue

            # Child is confirmed by ontology inference and the agent
            self.logger.debug(
                f"Adding {child.sctid} {child.pt} as a new ancestor"
            )
            new_anchors.add(child.sctid)

        if not new_anchors:
            self.logger.debug("No new ancestors found")
            return False

        # Update the anchor set with the new one
        self.logger.debug(f"New ancestors: {new_anchors}")
        self.portrait.ancestor_anchors |= new_anchors
        return True


# https://stackoverflow.com/questions/28655198/best-way-to-display-logs-in-pyqt
class BouzygesLoggingSpace(logging.Handler):
    """\
A logging handler that outputs log records to a QListView widget.
"""

    max_records = 2000

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.widget = QtWidgets.QListView(*args, **kwargs)
        self.model = QtCore.QStringListModel()
        self.widget.setModel(self.model)
        self.widget.setWordWrap(True)
        self.widget.setAlternatingRowColors(True)
        self.widget.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.setFormatter(FORMATTER)

    def emit(self, record):
        msg = self.format(record)

        current_count = self.model.rowCount()
        self.model.insertRow(current_count)
        index = self.model.index(current_count)
        self.model.setData(index, msg)

        # Scroll to the top
        slider = self.widget.verticalScrollBar()
        if slider:
            slider.setValue(slider.maximum())

        # Limit the number of records
        if self.model.rowCount() > self.max_records:
            self.model.removeRow(0)


class EnvironmentVariableEditor(QtWidgets.QDialog):
    """\
A dialog to edit environment variables.
"""

    def __init__(
        self, parent: BouzygesWindow, variables: EnvironmentParameters
    ) -> None:
        super().__init__(parent)
        self.logger = parent.logger.getChild(self.__class__.__name__)
        self.variables = variables
        self.parent_window = parent

        self.setWindowTitle("Override Environment variables")
        self.__dict = {}
        master_layout = QtWidgets.QVBoxLayout()
        warning_label = QtWidgets.QLabel(
            "<b>Warning</b>: Environment variables will not persist between "
            "sessions. For security reasons, they are neither logged nor "
            "saved in config JSON file. Use <code>.env</code> file for "
            "permanent changes."
        )
        warning_label.setWordWrap(True)
        master_layout.addWidget(warning_label)

        for envvar, value in variables.model_dump().items():
            self.__dict[envvar] = value
            layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(envvar.upper())
            label.setFixedWidth(200)
            layout.addWidget(label)
            edit = QtWidgets.QLineEdit()
            edit.setText(value)
            edit.setPlaceholderText("Unset")
            edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.PasswordEchoOnEdit)
            edit.setMinimumWidth(300)
            layout.addWidget(edit)
            master_layout.addLayout(layout)
        self.setLayout(master_layout)

        buttons_layout = QtWidgets.QHBoxLayout()
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save)

        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(save_button)
        master_layout.addLayout(buttons_layout)

    def save(self):
        for envvar, value in self.__dict.items():
            if value:
                logging.debug(f"Setting {envvar} to a new value")
            else:
                logging.debug(f"Unsetting {envvar}")
            setattr(self.variables, envvar, value or None)
            # Set them to os module to make sure they are available
            os.environ[envvar] = value or ""
        self.parent_window.reset_ui()
        self.accept()


class BouzygesWindow(QtWidgets.QMainWindow):
    """\
Main window and start config for the Bouzyges system.
"""

    def __init__(self, loop: asyncio.AbstractEventLoop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = LOGGER.getChild("GUI")
        self.setWindowTitle("OHDSI Bouzyges")
        self.setWindowIcon(QtGui.QIcon("icon.png"))

        layout = QtWidgets.QVBoxLayout()
        self.__populate_layout(layout)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Menu bar
        menubar = self.menuBar()
        if menubar is not None:
            self.populate_menu(menubar)

        # Async support
        self.loop = loop

    def __populate_layout(self, layout):
        self._vertical_spacer = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._fixed_vertical_spacer = QtWidgets.QSpacerItem(
            20,
            20,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        # Options
        self.options_container = QtWidgets.QFrame()
        self.options_container.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
        self.options_container.setMaximumWidth(350)
        options_layout = QtWidgets.QVBoxLayout()
        options_subtitle = QtWidgets.QLabel("Options")
        options_subtitle.setStyleSheet("font-weight: bold;")
        options_layout.addWidget(options_subtitle)
        options_contents = QtWidgets.QVBoxLayout()
        self.__populate_option_contents(options_contents)
        options_layout.addLayout(options_contents)
        options_layout.addItem(self._vertical_spacer)
        self.options_container.setLayout(options_layout)

        # Input and output
        right_quarter_layout = QtWidgets.QVBoxLayout()
        self.io_container = QtWidgets.QWidget()
        io_layout = QtWidgets.QVBoxLayout()
        self.__populate_io_layout(io_layout)
        self.io_container.setLayout(io_layout)
        right_quarter_layout.addWidget(self.io_container)
        right_quarter_layout.addItem(self._vertical_spacer)

        # Run layout
        run_layout = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setMaximumWidth(150)
        self.run_button.clicked.connect(self.spin_bouzyges)
        self.run_status = QtWidgets.QLabel("Status: Ready")
        self.run_status.setStyleSheet("font-weight: bold;")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setDisabled(True)

        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.run_status)
        right_quarter_layout.addLayout(run_layout)
        right_quarter_layout.addWidget(self.progress_bar)
        # Logging space
        logging_layout = QtWidgets.QVBoxLayout()
        logging_subtitle = QtWidgets.QLabel("Run log")
        logging_subtitle.setStyleSheet("font-weight: bold;")
        self.log_display = BouzygesLoggingSpace()

        LOGGER.addHandler(self.log_display)

        self.logger.info("Logging to GUI is initialized")
        log_widget = self.log_display.widget
        logging_layout.addWidget(logging_subtitle)
        logging_layout.addWidget(log_widget)

        top_half_layout = QtWidgets.QHBoxLayout()
        top_half_layout.addWidget(self.options_container)
        top_half_layout.addLayout(right_quarter_layout)

        layout.addLayout(top_half_layout)
        layout.addLayout(logging_layout)

    def __populate_io_layout(self, layout) -> None:
        # Input file selection
        input_frame = QtWidgets.QFrame()
        input_frame.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
        input_layout = QtWidgets.QVBoxLayout()
        input_subtitle = QtWidgets.QLabel("Input file")
        input_subtitle.setStyleSheet("font-weight: bold;")
        input_layout.addWidget(input_subtitle)
        input_doc = QtWidgets.QLabel()
        input_doc.setText(FileReader.input_doctext)
        input_layout.addWidget(input_doc)
        input_contents = QtWidgets.QHBoxLayout()
        self.input_file = QtWidgets.QLineEdit()
        self.input_file.setPlaceholderText("Select input CSV file")
        self.input_file.setText(PARAMS.read.file)
        self.input_file.setReadOnly(True)
        input_contents.addWidget(self.input_file)
        input_select = QtWidgets.QPushButton("Select")
        input_select.clicked.connect(self.select_input)
        input_contents.addWidget(input_select)
        input_layout.addLayout(input_contents)
        self.input_options_widget = IOParametersWidget(PARAMS.read, "Read")
        input_layout.addWidget(self.input_options_widget)
        input_frame.setLayout(input_layout)

        # Output selection
        output_frame = QtWidgets.QFrame()
        output_frame.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
        output_layout = QtWidgets.QVBoxLayout()
        output_subtitle = QtWidgets.QLabel("Output file")
        output_subtitle.setStyleSheet("font-weight: bold;")
        output_layout.addWidget(output_subtitle)
        out_dir_contents = QtWidgets.QHBoxLayout()
        out_dir_label = QtWidgets.QLabel("Output directory:")
        out_dir_contents.addWidget(out_dir_label)
        self.output_dir = QtWidgets.QLineEdit()
        self.output_dir.setPlaceholderText("Select output directory")
        self.output_dir.setText(os.getcwd())
        self.output_dir.setReadOnly(True)
        out_dir_contents.addWidget(self.output_dir)
        output_select = QtWidgets.QPushButton("Select")
        output_select.clicked.connect(self.select_output)
        out_dir_contents.addWidget(output_select)
        output_layout.addLayout(out_dir_contents)

        out_file_contents = QtWidgets.QHBoxLayout()
        out_file_label = QtWidgets.QLabel("File name:")
        out_file_contents.addWidget(out_file_label)
        output_filename = QtWidgets.QLineEdit()
        self.out_options_widget = IOParametersWidget(PARAMS.write, "Write")
        output_filename.setPlaceholderText("Output file name")
        output_filename.setText(PARAMS.write.file)
        output_filename.textChanged.connect(self.out_options_widget.update_file)
        out_file_contents.addWidget(output_filename)
        out_format_label = QtWidgets.QLabel("Format:")
        out_file_contents.addWidget(out_format_label)
        out_format_select = QtWidgets.QComboBox()
        out_format_select.addItems(FileWriter.get_formats())
        out_format_select.setCurrentIndex(
            FileWriter.get_formats().index(PARAMS.format)
        )
        out_format_select.currentIndexChanged.connect(self.format_changed)
        self.out_options_widget.setEnabled(PARAMS.format != "JSON")
        out_file_contents.addWidget(out_format_select)

        output_layout.addLayout(out_dir_contents)
        output_layout.addLayout(out_file_contents)
        output_layout.addWidget(self.out_options_widget)
        output_frame.setLayout(output_layout)

        layout.addWidget(input_frame)
        layout.addItem(self._vertical_spacer)
        layout.addWidget(output_frame)
        layout.addItem(self._vertical_spacer)

    def format_changed(self, idx: int) -> None:
        PARAMS.format = FileWriter.get_formats()[idx]
        self.out_options_widget.setEnabled(PARAMS.format != "JSON")
        self.logger.info(f"Output format changed to {PARAMS.format}")

    def select_input(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select input CSV file",
        )
        if file:
            self.input_options_widget.update_file(
                file[0], self.input_file.setText
            )

    def select_output(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select directory for output",
        )
        if dir:
            self.output_dir.setText(dir)
            PARAMS.out_dir = dir
            self.logger.info(f"Output directory changed to {dir}")

    def select_cache_db(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select cache database file",
        )
        if file:
            PARAMS.api.cache_db = file[0]

    @qasync.asyncSlot()
    async def spin_bouzyges(self) -> None:
        # Adding a file handler to the logger
        if PARAMS.log.log_to_file:
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f"bouzyges-{date_str}.log"
            log_file = os.path.join(self.output_dir.text(), file_name)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(FORMATTER)

            LOGGER.addHandler(file_handler)
            self.logger.info(f"Now logging to file: {log_file}")

        self.options_container.setEnabled(False)
        self.io_container.setEnabled(False)
        self.run_button.setEnabled(False)
        bouzyges_capture: dict[Literal["success"], Bouzyges] = {}
        bouzyges: Bouzyges

        async def prepare(progress_callback):
            try:
                bouzyges_capture["success"] = await Bouzyges.prepare(
                    progress_callback=progress_callback
                )
            except Exception as e:
                self.logger.error(f"Could not prepare Bouzyges: {e}")
                self.logger.error(
                    "Could not prepare Bouzyges. Check the configuration."
                )
                self.reset_ui(fail=True)
                return

        await self.__start_job("Preparation", prepare)
        bouzyges = bouzyges_capture["success"]

        run_success = await self.__start_job("Run", bouzyges.run)
        if not run_success:
            self.reset_ui(fail=True)
            return

    def reset_ui(self, fail=False) -> None:
        self.input_options_widget.set_values()
        self.out_options_widget.set_values()
        self.options_container.setEnabled(True)
        self.io_container.setEnabled(True)
        self.run_button.setEnabled(True)
        self.run_status.setText(
            "Status: Error occured" if fail else "Status: Ready"
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setEnabled(False)
        self.progress_bar.setTextVisible(False)

    def update_progress(self, value: int, max: int) -> None:
        self.progress_bar.setRange(0, max)
        self.progress_bar.setValue(value)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat(f"{value}/{max}")

    def reset_progress(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFormat("")

    async def __start_job(self, name: str, async_target, *args, **kwargs):
        self.logger.info(f"Starting {name} thread")
        self.run_status.setText(f"Status: Bouzy ({name})")
        self.progress_bar.setEnabled(True)
        result = await async_target(
            *args, **kwargs, progress_callback=self.update_progress
        )
        self.reset_progress()
        self.logger.info(f"{name} thread finished")
        return result

    def __populate_option_contents(self, layout) -> None:
        # Prompter choice:
        prompter_layout = QtWidgets.QVBoxLayout()

        ## Prompter imlementations
        impl_layout = QtWidgets.QHBoxLayout()
        impl_label = QtWidgets.QLabel("Prompter:")
        impl_label.setStyleSheet("font-weight: bold;")
        impl_options = QtWidgets.QComboBox()
        impl_options.addItems(AVAILABLE_PROMPTERS.values())
        current_impl_idx = list(AVAILABLE_PROMPTERS).index(PARAMS.api.prompter)
        impl_options.setCurrentIndex(current_impl_idx)
        impl_options.currentIndexChanged.connect(self.prompter_changed)
        impl_layout.addWidget(impl_label)
        impl_layout.addWidget(impl_options)

        ## Model id
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model ID:")
        model_input = QtWidgets.QLineEdit()
        model_input.setPlaceholderText(DEFAULT_MODEL)
        model_input.setText(PARAMS.api.llm_model_id)
        model_input.textChanged.connect(self.model_id_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_input)

        ## Prompt repetition
        repeat_layout = QtWidgets.QHBoxLayout()
        repeat_label = QtWidgets.QLabel("<u>Repeat each prompt:</u>")
        repeat_label.setToolTip(
            "Number of times to repeat each prompt for the LLM. The final "
            "output will be the best result of all repetitions, e.g. for 5 "
            "repetitions queries will stop after getting 3 of the same results."
        )
        repeat_input = QtWidgets.QLineEdit()
        repeat_input.setPlaceholderText(f"Default: {DEFAULT_REPEAT_PROMPTS}")
        repeat_text = (
            "" if (rp := PARAMS.api.repeat_prompts) is None else str(rp)
        )
        repeat_input.setText(repeat_text)
        repeat_input.textChanged.connect(self.repeat_prompts_changed)
        repeat_layout.addWidget(repeat_label)
        repeat_layout.addWidget(repeat_input)

        prompter_layout.addLayout(impl_layout)
        prompter_layout.addLayout(model_layout)
        prompter_layout.addLayout(repeat_layout)

        # Snowstorm connection options
        snowstorm_layout = QtWidgets.QVBoxLayout()
        snowstorm_subtitle = QtWidgets.QLabel("Snowstorm API:")
        snowstorm_subtitle.setStyleSheet("font-weight: bold;")
        snowstorm_layout.addWidget(snowstorm_subtitle)
        snowstorm_contents = QtWidgets.QVBoxLayout()
        snowstorm_url_layout = QtWidgets.QHBoxLayout()
        snowstorm_url_label = QtWidgets.QLabel("Endpoint URL:")
        snowstorm_url_input = QtWidgets.QLineEdit()
        snowstorm_url_input.setPlaceholderText("http://localhost:8080/")
        snowstorm_url_input.setText(PARAMS.api.snowstorm_url)
        snowstorm_url_input.textChanged.connect(self.snowstorm_url_changed)
        snowstorm_url_layout.addWidget(snowstorm_url_label)
        snowstorm_url_layout.addWidget(snowstorm_url_input)
        snowstorm_contents.addLayout(snowstorm_url_layout)
        snowstorm_layout.addLayout(snowstorm_contents)

        # Sqlite database options
        sqlite_layout = QtWidgets.QVBoxLayout()
        sqlite_subtitle = QtWidgets.QLabel(
            "Prompt cache path (set to empty to disable):"
        )
        sqlite_subtitle.setStyleSheet("font-weight: bold;")
        sqlite_db_file = QtWidgets.QLineEdit()
        sqlite_db_file.setPlaceholderText("None")
        sqlite_db_file.setText(PARAMS.api.cache_db)
        sqlite_db_file.textChanged.connect(self.cache_db_changed)
        sqlite_select = QtWidgets.QPushButton("Select")
        sqlite_select.clicked.connect(self.select_cache_db)
        sqlite_layout.addWidget(sqlite_subtitle)
        sqlite_selector_layout = QtWidgets.QHBoxLayout()
        sqlite_selector_layout.addWidget(sqlite_db_file)
        sqlite_selector_layout.addWidget(sqlite_select)
        sqlite_layout.addLayout(sqlite_selector_layout)

        # Concurrent workers
        concurrent_layout = QtWidgets.QHBoxLayout()
        concurrent_label = QtWidgets.QLabel("Max concurrent workers:")
        concurrent_input = QtWidgets.QLineEdit()
        concurrent_input.setPlaceholderText("No concurrency")
        concurrent_input.setText(str(PARAMS.api.max_concurrent_workers))
        concurrent_input.textChanged.connect(self.concurrent_workers_changed)
        concurrent_layout.addWidget(concurrent_label)
        concurrent_layout.addWidget(concurrent_input)

        # Developer options
        prof_label = QtWidgets.QLabel("Profiling:")
        prof_label.setStyleSheet("font-weight: bold;")
        prof_layout = QtWidgets.QHBoxLayout()

        early_termination_label = QtWidgets.QLabel("Stop after (s):")
        early_termination_input = QtWidgets.QLineEdit()
        early_termination_input.setPlaceholderText("don't")
        if PARAMS.prof.stop_profiling_after_seconds is not None:
            early_termination_input.setText(
                str(PARAMS.prof.stop_profiling_after_seconds)
            )
        early_termination_input.textChanged.connect(self.et_changed)

        profiling_layout = QtWidgets.QVBoxLayout()
        profiling_checkbox = QtWidgets.QCheckBox("Generate stats.prof")
        profiling_checkbox.setChecked(PARAMS.prof.enabled)
        profiling_checkbox.stateChanged.connect(self.profiling_changed)
        profiling_layout.addWidget(profiling_checkbox)

        prof_layout.addWidget(early_termination_label)
        prof_layout.addWidget(early_termination_input)
        prof_layout.addLayout(profiling_layout)

        logging_label = QtWidgets.QLabel("Logging:")
        logging_label.setStyleSheet("font-weight: bold;")
        logging_layout = QtWidgets.QHBoxLayout()

        log_to_file_checkbox = QtWidgets.QCheckBox("Log to file")
        log_to_file_checkbox.setChecked(PARAMS.log.log_to_file)
        log_to_file_checkbox.stateChanged.connect(self.ltf_changed)

        logging_level_label = QtWidgets.QLabel("Logging level:")
        logging_level_options = QtWidgets.QComboBox()
        logging_level_options.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        current_logging_level_idx = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ].index(PARAMS.log.logging_level)
        logging_level_options.setCurrentIndex(current_logging_level_idx)
        logging_level_options.currentIndexChanged.connect(self.ll_changed)

        logging_layout.addWidget(logging_level_label)
        logging_layout.addWidget(logging_level_options)
        logging_layout.addWidget(log_to_file_checkbox)

        for child in [
            prompter_layout,
            snowstorm_layout,
            sqlite_layout,
            concurrent_layout,
        ]:
            layout.addLayout(child)
            layout.addItem(self._fixed_vertical_spacer)

        layout.addWidget(prof_label)
        layout.addLayout(prof_layout)
        layout.addItem(self._fixed_vertical_spacer)
        layout.addWidget(logging_label)
        layout.addLayout(logging_layout)

    def ltf_changed(self, state) -> None:
        PARAMS.log.log_to_file = state == 2
        self.logger.debug(f"Logging to file changed to: {state == 2}")

    def prompter_changed(self, index) -> None:
        new_prompter = list(AVAILABLE_PROMPTERS)[index]
        PARAMS.api.prompter = new_prompter
        self.logger.debug(f"Prompter changed to: {new_prompter}")

    def profiling_changed(self, state) -> None:
        PARAMS.prof.enabled = state == 2
        self.logger.debug(f"Profiling changed to: {state == 2}")

    def et_changed(self, text) -> None:
        try:
            new_et = int(text)
        except ValueError:
            PARAMS.prof.stop_profiling_after_seconds = None
            self.logger.warning(
                "Invalid input for early termination, disabling"
            )
            return

        if new_et <= 0:
            new_et = None

        PARAMS.prof.stop_profiling_after_seconds = new_et
        self.logger.debug(f"Early termination changed to: {new_et}")

    def repeat_prompts_changed(self, text) -> None:
        try:
            new_repeat = int(text)
        except ValueError:
            PARAMS.prof.stop_profiling_after_seconds = None
            self.logger.debug("Invalid input for prompt repeats, resetting")
            return

        if new_repeat <= 1:
            # Make sure it's at least 1
            self.logger.warning("Prompt repetition must be at least 1")
            new_repeat = 1

        PARAMS.api.repeat_prompts = new_repeat
        self.logger.debug(f"Prompt repetition set to: {new_repeat}")

    def concurrent_workers_changed(self, text) -> None:
        try:
            new_workers = int(text)
        except ValueError:
            PARAMS.api.max_concurrent_workers = 1
            self.logger.debug("Invalid input for workers, disabling")
            return

        if new_workers <= 1:
            self.logger.warning("Concurrency implicitly disabled")
            new_workers = 1

        PARAMS.api.max_concurrent_workers = new_workers
        self.logger.debug(f"Max concurrent workers set to: {new_workers}")

    def ll_changed(self, index) -> None:
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
        new_level = levels[index]
        self.logger.debug(f"Logging level changed to: {new_level}")
        PARAMS.log.update(new_level)

    def snowstorm_url_changed(self, text) -> None:
        if not text:
            text = "http://localhost:8080/"
        PARAMS.api.snowstorm_url = Url(text)
        self.logger.debug(f"Snowstorm URL changed to: {text}")

    def model_id_changed(self, text) -> None:
        if not text:
            text = DEFAULT_MODEL

        PARAMS.api.llm_model_id = text
        self.logger.debug(f"LLM model_id set to: {text}")

    def cache_db_changed(self, text) -> None:
        if not text:
            text = None
        PARAMS.api.cache_db = text
        self.logger.debug(f"Cache database path changed to: {text}")

    def populate_menu(self, menubar: QtWidgets.QMenuBar) -> None:
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        help_menu = menubar.addMenu("Help")

        if file_menu is None or edit_menu is None or help_menu is None:
            raise RuntimeError("Could not create menu bar")

        # File menu items
        load_config_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("document-open"),
            text="Load configuration",
            parent=self,
        )
        load_config_action.triggered.connect(self.load_config)
        load_config_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        file_menu.addAction(load_config_action)

        save_config_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("document-save"),
            text="Save configuration",
            parent=self,
        )
        save_config_action.triggered.connect(self.save_config)
        save_config_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        file_menu.addAction(save_config_action)

        quit_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("application-exit"),
            text="Quit",
            parent=self,
        )
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu items
        edit_env_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("preferences-system"),
            text="Override environment variables",
            parent=self,
        )
        edit_env_action.triggered.connect(
            lambda: EnvironmentVariableEditor(self, PARAMS.env).exec()
        )
        edit_menu.addAction(edit_env_action)

        # Help menu items
        about_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("help-about"), text="About", parent=self
        )
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

        license_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("help-about"),
            text="License",
            parent=self,
        )
        license_action.triggered.connect(self.license)
        help_menu.addAction(license_action)

        report_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("help-report-bug"),
            text="Report an issue or get help",
            parent=self,
        )
        report_action.triggered.connect(self.report_issue)
        report_action.setShortcut(QtGui.QKeySequence.StandardKey.HelpContents)
        help_menu.addAction(report_action)

    def about(self):
        label = (
            "Bouzyges is a tool for identifying the most specific "
            "ancestors of a set of terms in the SNOMED CT ontology."
            "\n\n"
            "All information is available at the project's GitHub page."
        )
        QtWidgets.QMessageBox.about(self, "About Bouzyges", label)

    def report_issue(self):
        webbrowser.open("https://github.com/OHDSI/Bouzyges")

    def license(self):
        QtWidgets.QMessageBox.about(
            self,
            "License",
            """\
Copyright ©️ 2024 Eduard Korchmar, EPAM Systems and OHDSI community

This program is free software: you can redistribute it and/or modify \
it under the terms of the GNU General Public License as published by \
the Free Software Foundation, either version 3 of the License, or \
(at your option) any later version.

This program is distributed in the hope that it will be useful, \
but WITHOUT ANY WARRANTY; without even the implied warranty of \
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the \
GNU General Public License for more details.

You should have received a copy of the GNU General Public License \
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Bouzyges logo is generated by DALL-E model by OpenAI and is not copyrightable.
""",
        )

    def load_config(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select JSON configuration file",
        )
        if file:
            with open(file[0], "r") as f:
                json_config = json.load(f)
                PARAMS.update(json_config)
                self.reset_ui()

            self.logger.info(f"Configuration loaded from {file[0]}")

    def save_config(self):
        file = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save JSON configuration file",
        )
        if file:
            with open(file[0], "w") as f:
                json.dump(PARAMS.model_dump(), f, indent=2)

            self.logger.info(f"Configuration saved to {file[0]}")


def main():
    loop = qasync.QEventLoop(APP)
    asyncio.set_event_loop(loop)
    window = BouzygesWindow(loop=loop)
    window.show()
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    APP = qasync.QApplication(sys.argv)
    main()
