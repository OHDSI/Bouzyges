-- Create a table to store prompts and responses
CREATE TABLE prompt (
  model TEXT NOT NULL,
  prompt TEXT NOT NULL,
  api_options TEXT,
  response TEXT NOT NULL,
  PRIMARY KEY (model, prompt, api_options)
);

