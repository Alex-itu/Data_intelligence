# from plotnine import ggplot, aes, geom_point
# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 7, 11]})
# plot = ggplot(df, aes(x='x', y='y')) + geom_point()

# # Convert to a Matplotlib figure and show
# fig = plot.draw()
# plt.show()



from plotnine import ggplot, aes, geom_point, ggsave
import pandas as pd

# Sample data
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 7, 11]})

# Create plot
plot = ggplot(df, aes(x='x', y='y')) + geom_point()

# Show in a separate window
ggsave(plot=plot, filename="dummy.png", show=True)
plot.show(plot)





# from plotnine import ggplot, aes, geom_point
# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample Data
# df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 7, 11]})

# # Create Plot
# plot = ggplot(df, aes(x='x', y='y')) + geom_point()

# # Convert plotnine object to a Matplotlib figure
# fig = plot.draw()

# # Show the plot in a separate window
# plt.show()