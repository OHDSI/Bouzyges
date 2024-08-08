-- Create a table to store prompts and responses
CREATE TABLE prompt (
  model TEXT NOT NULL,
  prompt_text TEXT NOT NULL,
  prompt_is_json BOOLEAN NOT NULL DEFAULT FALSE,
  api_options TEXT,
  response TEXT NOT NULL,
  PRIMARY KEY (model, prompt_text, prompt_is_json, api_options)
);

