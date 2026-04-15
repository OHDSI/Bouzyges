import json
import sqlite3

from prompt import Prompt
from utils.logger import LOGGER


class PromptCache:
    """\
Interface for a prompt cache.

Saves prompts and answers to avoid re-prompting the same questions and wasting
tokens.
"""

    def __init__(self, db_connection: sqlite3.Connection):
        # TODO: form an event queue for this; sqlite does not do well in
        # multi-threaded environments
        self.connection = db_connection
        self.table_name = "prompt"

        self.logger = LOGGER.getChild("PromptCache")

        # Create the table if it does not exist in DB
        table_exists_query = """\
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name=?;
        """
        exists = self.connection.execute(table_exists_query, [self.table_name])
        if not exists.fetchone():
            self.logger.info("Creating prompt cache table")
            with open("init_prompt_cache.sql") as f:
                self.connection.executescript(f.read())
                self.connection.commit()
        else:
            self.logger.info("Existing prompt table already exists")

    def get(self, model: str, prompt: Prompt, attempt: int) -> str | None:
        """\
Get the answer from the cache for specified model.
"""
        prompt_dict = prompt.to_json()
        api_are_none = prompt_dict["api_options"] is None

        query = f"""
            SELECT response
            FROM {self.table_name}
            WHERE
                attempt = ? AND
                model = ? AND
                prompt_text = ? AND
                prompt_is_json = ? AND
                api_options {"IS" if api_are_none else "="} ?
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                query,
                (attempt, model, *prompt_dict.values()),
            )
            if answer := cursor.fetchone():
                return answer[0]
            return None
        except sqlite3.InterfaceError, sqlite3.DatabaseError:
            self.logger.warning(
                "Cache access failed for prompt: "
                + f"{json.dumps(prompt.to_json())}"
            )
            return None

    def remember(
        self, model: str, prompt: Prompt, response: str, attempt: int
    ) -> None:
        """\
Remember the answer for the prompt for the specified model.
"""
        query = f"""
            INSERT INTO {self.table_name} (
                attempt,
                model,
                prompt_text,
                prompt_is_json,
                api_options,
                response
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """
        # Convert prompt to serializable format
        prompt_dict = prompt.to_json()

        cursor = self.connection.cursor()
        cursor.execute(
            query,
            (
                attempt,
                model,
                *prompt_dict.values(),
                response,
            ),
        )
        self.connection.commit()

    @classmethod
    def use_db(cls, db_path: str) -> PromptCache:
        """\
Create a PromptCache instance using a SQLite database at the specified path.
"""
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return PromptCache(conn)
