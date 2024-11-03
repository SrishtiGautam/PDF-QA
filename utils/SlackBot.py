from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class SlackBot:
    """Class responsible for posting messages to Slack."""

    def __init__(self, cfg, logger):
        self.logger = logger
        self.client = WebClient(token=cfg.slack_token)
        self.channel = cfg.slack_channel

    def post_message(self, text: str):
        """Posts a message to the specified Slack channel."""
        try:
            self.client.chat_postMessage(channel=self.channel, text=text)
        except SlackApiError as e:
            self.logger.error(f"Slack API error: {e.response['error']}")
            raise