# Week 5 exercises: Neural Networks and Backpropagation

Run the original script first and save the terminal output. Change one variable at a time.

1. **Learning rate:** Change `LEARNING_RATE` from `0.1` to `0.01`. Does the loss fall as quickly? Why?
2. **Training duration:** Set `EPOCHS = 100`. Compare predictions with the 3,000-epoch run.
3. **Hidden-layer width:** Change `nn.Linear(2, 4)` to `nn.Linear(2, 2)` and the output layer to `nn.Linear(2, 1)`. Does this seeded run converge?
4. **Nonlinearity:** Replace `nn.Tanh()` with `nn.Identity()`. Can the resulting linear network learn XOR correctly? Explain why.
5. **Inspect gradients:** Immediately after `loss.backward()`, print `model[0].weight.grad` during epoch 1. How do gradients differ from weights?
6. **Trace the update:** Print a chosen weight immediately before and after `optimizer.step()` in epoch 1. Which operation changes the weight?

**Reflection:** Explain the difference between forward propagation, backpropagation, and an optimizer update in your own words. Do not interpret training accuracy on four examples as evidence of generalization.
