def train_epoch(model, loader, plot_loss, print_loss):
  model.train()

  LOSS = 0
  plot = []

  for i, (input, target) in enumerate(loader):
    optimizer.zero_grad()

    logits, _ = model(input)

    loss = criterion(logits, target)
    LOSS += loss.item()
    loss.backward()

    optimizer.step()

    if (i+1)%print_loss == 0:
      print(f'Training epoch {i+1}/{len(loader)}: {loss.item():.5f}')

    if i%plot_loss == 0:
      plot.append(loss.item())

  LOSS /= len(loader)
  return LOSS, plot

def get_time(epoch_time):
  minutes = int(epoch_time) // 60
  seconds = epoch_time - minutes*60
  return f'Time taken: {minutes} m. {seconds:.1f} s.'
