from openai import OpenAI
import time

class LModel:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.client = OpenAI(api_key=cfg.api_key)
        self.model = cfg.model
        self.embed_model = cfg.embed_model
        self.confidence_threshold = cfg.confidence_threshold
        self.max_retries = 3
        self.temparature = 0

    def embed_chunk(self, chunk):
        """Call embedding model to embed the text chunks"""
        attempts = 0
        while attempts < self.max_retries:
            try:
                response = self.client.embeddings.create(
                    input=chunk,
                    model=self.embed_model
                )
                return response.data[0].embedding
            except Exception as e:
                attempts += 1
                self.logger.error(f"Error embedding chunk on attempt {attempts}: {e}")

                if attempts == self.max_retries:
                    self.logger.error("Max retries reached for embedding chunk. Returning None.")
                    return None  # Or handle as needed
                else:
                    wait_time = 2 ** attempts  # Exponential backoff
                    self.logger.info(f"Retrying embedding in {wait_time} seconds...")
                    time.sleep(wait_time)  # Wait before retrying


    def is_low_confidence(self, log_probs):
        """Determine if the answer has low confidence based on log probabilities."""
        average_log_prob = 0
        for item in log_probs:
            average_log_prob += item.logprob

        average_log_prob/=len(log_probs)

        # Compare the average log probability to the confidence threshold
        return average_log_prob < self.confidence_threshold


    def low_confidence_phrases(self, answer):
        """Check if the answer is low-confidence."""
        low_confidence_phrases = ["I am not sure", "possibly", "maybe", "not certain","not available", "no information available", "no available information"]
        for phrase in low_confidence_phrases:
            if phrase in answer.lower():
                return True
        return False


    def query_model(self, query, contexts):
        """Find answers to the query based on given contexts"""
        response_text = []
        for chunk in contexts:
            attempts = 0
            while attempts < self.max_retries:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages = [
                            {"role": "user", "content": f"Context: {chunk}\n\nQuestion: {query}\nAnswer:"}
                        ],
                        max_tokens=200,
                        logprobs=True,
                        temperature=self.temparature
                    )
                    response_content = response.choices[0].message.content.strip()
                    log_probs = response.choices[0].logprobs.content
                    if self.is_low_confidence(log_probs):
                        response_content = "Data Not Available"

                    response_text.append(response_content)
                    break

                except Exception as e:
                    attempts += 1
                    self.logger.error(f"Error querying model on attempt {attempts} for chunk: {chunk}. Error: {e}")

                    if attempts == self.max_retries:
                        self.logger.error(f"Max retries reached for chunk: {chunk}. Returning 'Data Not Available'.")
                        response_text.append("Data Not Available")
                    else:
                        wait_time = 2 ** attempts  # Exponential backoff
                        self.logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)  # Wait before retrying

        # Combine responses into one coherent answer
        combined_response = "\n".join(response_text).strip()
        return self.final_answer(combined_response)


    def final_answer(self, answers):
        """Combine answers from all contexts by utilizing LLM"""

        final_prompt = f"Based on the following answers, please provide a final, concise answer:\n{''.join(answers)}"
        attempts = 0
        while attempts < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": final_prompt}
                    ],
                    max_tokens=200,
                    temperature=self.temparature
                )
                answer = response.choices[0].message.content.strip()
                if self.low_confidence_phrases(answer):
                    answer = "Data Not Available"

                return answer

            except Exception as e:
                attempts += 1
                self.logger.error(f"Error generating final answer on attempt {attempts}. Error: {e}")

                if attempts == self.max_retries:
                    self.logger.error("Max retries reached for final answer generation.")
                    return "Data Not Available"
                else:
                    wait_time = 2 ** attempts  # Exponential backoff
                    self.logger.info(f"Retrying final answer generation in {wait_time} seconds...")
                    time.sleep(wait_time)  # Wait before retrying

