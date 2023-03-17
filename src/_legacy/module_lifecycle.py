"""def fit(self): if global_rank == 0:

        # prepare data is called on GLOBAL_ZERO only
        prepare_data()

    configure_callbacks()

    with parallel(devices):
        # devices can be GPUs, TPUs, ...
        train_on_device(model)


def train_on_device(model):
    # called PER DEVICE
    setup("fit")
    configure_optimizers()
    on_fit_start()

    # the sanity check runs here

    on_train_start()
    for epoch in epochs:
        fit_loop()
    on_train_end()

    on_fit_end()
    teardown("fit")


def fit_loop():
    on_train_epoch_start()

    for batch in train_dataloader():
        on_train_batch_start()

        on_before_batch_transfer()
        transfer_batch_to_device()
        on_after_batch_transfer()

        training_step()

        on_before_zero_grad()
        optimizer_zero_grad()

        on_before_backward()
        backward()
        on_after_backward()

        on_before_optimizer_step()
        configure_gradient_clipping()
        optimizer_step()

        on_train_batch_end()

        if should_check_val:
            val_loop()
    # end training epoch
    training_epoch_end()

    on_train_epoch_end()


def val_loop():
    on_validation_model_eval()  # calls `model.eval()`
    torch.set_grad_enabled(False)

    on_validation_start()
    on_validation_epoch_start()

    val_outs = []
    for batch_idx, batch in enumerate(val_dataloader()):
        on_validation_batch_start(batch, batch_idx)

        batch = on_before_batch_transfer(batch)
        batch = transfer_batch_to_device(batch)
        batch = on_after_batch_transfer(batch)

        out = validation_step(batch, batch_idx)

        on_validation_batch_end(batch, batch_idx)
        val_outs.append(out)

    validation_epoch_end(val_outs)

    on_validation_epoch_end()
    on_validation_end()

    # set up for train
    on_validation_model_train()  # calls `model.train()`
    torch.set_grad_enabled(True)
"""
