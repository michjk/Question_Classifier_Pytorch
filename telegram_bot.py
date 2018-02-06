from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import requests

import argparse

parser = argparse.ArgumentParser(description="This is for establishing telegram bot")
parser.add_argument('-t', '--token', help="telegram token", required=True)

args = parser.parse_args()

#'543323547:AAHAEgEUZYRXKnMg-lUEcwv0TLxDXJ2Rib8'
TOKEN = args.token
URL = 'http://155.69.146.216:4444'

def predict(bot, update):
    predict_msg = {'question': update.message.text}
    resp = requests.post(URL+"/predict", json=predict_msg)
    update.message.reply_text("Possible topic: " + resp.json()['result'])

def main():
    # Create Updater object and attach dispatcher to it
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    print("Bot started")

    # Add command handler to dispatcher
    predict_handler = MessageHandler(Filters.text, predict)
    dispatcher.add_handler(predict_handler)

    # Start the bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C
    updater.idle()

if __name__ == '__main__':
    main()

