import settings
settings.load()
import logging

def on_validation_start():
    logging.info("Validation...")
    logging.info("-"*10)


def on_validation_end(*metrics):
    res = f"Validation loss: ["
    res += "".join(f"({metric.title}): {metric.item():.4f}|" for metric in metrics)
    res += "]"
    logging.info(res)

def on_step_end(step, **kwargs):
    pass
def on_epoch_start(epoch, **kwargs):
    logging.info(f"Epoch [{epoch+1}/{settings.ARGS['epochs']}]")

def on_epoch_end(epoch, **kwargs):
    res = f"End of epoch {epoch+1}"
    res += "".join(f",{key}:{value:.4f}" for key,value in kwargs.items())
    res += "."
    logging.info(res)
    logging.info("-"*30)

def on_batch_end(batch, **kwargs):
    res = f"End of batch {batch+1}"
    res += "".join(f",{key}:{value:.4f}" for key,value in kwargs.items())
    res += "."
    logging.info(res)
    logging.info("."*30)