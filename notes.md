# notes

### how to run the chess ui

`cd devtools/`
`npm run dev`

### pythonb backend entry point

root of the project
it's started when you run the next.js app

### bot logic

`src/main.py`
this is where our actual bot is implemented
ran by server.py

run
`python3 -m venv venv`
`./venv/scripts/activate` or `./venv/bin/activate`

```
# 1. serve.py receives:
{
  "pgn": "1. e4 e5 2. Nf3",
  "timeleft": 45000
}

# 2. chess_manager.set_context() reconstructs board from PGN

# 3. chess_manager.get_model_move() calls test_func(ctx)

# 4. test_func receives GameContext with:
#    - board: Board object at position after "2. Nf3"
#    - timeLeft: 45000
#    - logProbabilities: callback function

# 5. Bot returns Move object (e.g., Move.from_uci("Nc3"))

# 6. serve.py returns:
{
  "move": "b1c3",
  "move_probs": {"b1c3": 0.3, "d2d4": 0.2, ...},
  "time_taken": 150.5,
  "logs": "Cooking move...\n...",
  "error": null
}
```


1. TRAIN CHESS BOT
2. PLAY AGAINST OTHER CHESS BOTS


do we add everything into the github repo or just the bot (i think its just the bot)


deployments
deployment slots - versions of our bot that are playing games
games
