# -*- coding: utf-8 -*-
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

bot = ChatBot(
    'Django ChatterBot Example',
    storage_adapter="chatterbot.storage.MongoDatabaseAdapter",
    logic_adapters=['chatterbot.logic.BestMatch'], 
    database="trainSample"      
    )

bot.set_trainer(ChatterBotCorpusTrainer)
bot.train('chatterbot.corpus.chinese')