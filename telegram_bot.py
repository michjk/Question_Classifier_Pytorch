from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, ConversationHandler, MessageHandler, Filters

import requests
import logging
import argparse
import traceback

from google_sheet_api import send_log

parser = argparse.ArgumentParser(description="This is for establishing telegram bot")
parser.add_argument('-t', '--token', help="telegram token", required=True)
parser.add_argument('--ip', help="IP address", required = True)

args = parser.parse_args()

#'543323547:AAHAEgEUZYRXKnMg-lUEcwv0TLxDXJ2Rib8'
TOKEN = args.token
URL = args.ip

# http request logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#Next stage conversation
FEEDBACK = 0
RECOMMENDED = 1
cache_data = {}

def predict(bot, update):
    try:
        predict_msg = {'question': update.message.text}
        logger.debug("Message: " + update.message.text)
        resp = requests.get(URL+"/predict", params=predict_msg)
        logger.debug("Predicted: " + resp.json()['result'])
        update.message.reply_text("Possible topic: " + resp.json()['result'])

        question = predict_msg
        predicted = resp.json()['result']
        
        cache_data[update.message.chat_id] = []
        cache_data[update.message.chat_id].append(update.message.text)
        cache_data[update.message.chat_id].append(predicted)
        
        keyboard = [
            [
                InlineKeyboardButton(u"Yes", callback_data="yes"),
                InlineKeyboardButton(u"No", callback_data="no")
            ]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text("Do you satisfied with the answer?", reply_markup = reply_markup)
        
        return FEEDBACK
    except:
        logger.debug(traceback.format_exc())
        update.message.reply_text("Error, please repeat again!")

def feedback(bot, update):
    try:
        query = update.callback_query
        logger.debug("Satisfied: " + query.data)
        if (query.data == 'yes'):
            cache_data[query.message.chat_id].append("none")
            send_log(cache_data[query.message.chat_id])
            query.message.reply_text("Thanks for responding")
            return
        keyboard = [
                [InlineKeyboardButton(u"Academic", callback_data="Academic")],
                [InlineKeyboardButton(u"Admission & Financial Services", callback_data="Admission & Financial Services")],
                [InlineKeyboardButton(u"Campus Life & Accommodation", callback_data="Campus Life & Accommodation")],
                [InlineKeyboardButton(u"Outreach & Exchange", callback_data="Outreach & Exchange")],
                [InlineKeyboardButton(u"Other services", callback_data="Other services")]
            
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        query.message.reply_text("Choose your recommendation?", reply_markup = reply_markup) 

        return RECOMMENDED
    except:
        logger.debug(traceback.format_exc())
        query.message.reply_text("Error, please repeat again!")

def recommended(bot, update):
    query = update.callback_query
    logger.debug("Recommended: " + query.data)
    
    cache_data[query.message.chat_id].append(query.data)
    send_log(cache_data[query.message.chat_id])

    query.message.reply_text("Thanks for responding")
    return

def help_command(bot, update):
    try:
        update.message.reply_text("This is NTU FAQ topic classifier. Send a message and you get the predicted topic. The topic are:" \
            "\n1. Academic" \
            "\n2. Admission & Financial Services" \
            "\n3. Campus Life & Accommodation" \
            "\n4. Outreach & Exchange" \
            "\n5. Other services" \
            )
    except:
        logger.debug(traceback.format_exc())

def main():
    # Create Updater object and attach dispatcher to it
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    
    # Add command handler to dispatcher

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.text, predict)],
        states={
            FEEDBACK: [CallbackQueryHandler(feedback)],
            RECOMMENDED: [CallbackQueryHandler(recommended)]
        },
        fallbacks=[MessageHandler(Filters.text, predict)]
    )
    help_handler = CommandHandler('help', help_command)
    
    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(help_handler)

    # Start the bot
    updater.start_polling()
    logger.debug("Bot started!")

    # Run the bot until you press Ctrl-C
    updater.idle()

if __name__ == '__main__':
    main()
