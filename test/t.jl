using Lux

rnn = Lux.RNNCell(10 => 20)

x = randn(10)

x, h = rnn(x, ps, st)

ps, st = Lux.setup(rng, rnn);
ps1, st1 = Lux.setup(rng, rnn);

x, h = rnn(x, ps, st)
x, h = rnn(x, ps1, st1)


