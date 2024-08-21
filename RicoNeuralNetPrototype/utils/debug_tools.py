#!/usr/bin/env python3

"""
In this file, we have a collection of utils for visualizations:
- Cost Visualization
- Gradient visualization
"""
from collections import defaultdict, deque

import dash
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch
from dash import dcc, html
from torch.nn.modules import Module


class FCNDebuggerConfig:
    compute_forward_gradients = True
    _log_batch_norm_stats = True


class FCNDebugger:
    def __init__(
        self,
        model: torch.nn.Module,
        config: FCNDebuggerConfig,
        X_test,
        y_test,
        predict_func,
    ):
        self.model = model
        self.config = config
        self._losses = []
        self._gradient_norms = defaultdict(list)
        self._initial_weights = self._save_weights()
        self._plot_func_ids: list = [
            (self._plot_costs, "cost-plot"),
            (self._plot_grad_norms, "grad-norm-plot"),
            (self._plot_initial_statistics, "weights-distribution"),
            (self._plot_accuracy, "Test Set Accuracy"),
        ]
        self._X_test = X_test if type(X_test) == np.ndarray else X_test.detach().numpy()
        self._y_test = y_test if type(y_test) == np.ndarray else y_test.detach().numpy()
        self._predict_func = predict_func

    def record_and_calculate_backward_pass(self, loss):
        self._losses.append(loss.item())
        self._compute_gradient_norm()

    def _compute_gradient_norm(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                self._gradient_norms[name].append(grad_norm)

    def _store_activations(self):
        # counting percentage of zero (dead neurons) in the output. Helpful especially for ReLu Activations
        pass

    def _log_batch_norm_stats(self):
        pass

    def _log_softmax_inputs_outputs(self):
        pass

    def _save_weights(self):
        weights = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:  # This ensures we're only saving weights, not biases
                weights[name] = param.detach().cpu().numpy().flatten()
        return weights

    def _calculate_weight_statistics(self):
        # Calculate mean and variance for initial weights
        mean_variances = {}
        for layer_name, weights in self._initial_weights.items():
            mean_variances[layer_name] = {
                "mean": sum(weights) / len(weights),
                "variance": sum((x - sum(weights) / len(weights)) ** 2 for x in weights)
                / len(weights),
            }
        return mean_variances

    def _format_fig(self, fig, title, xaxis_title, yaxis_title, plot_bgcolor="beige"):
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            font=dict(family="Comic Sans MS", size=14, color="blue"),
            title_font=dict(family="Comic Sans MS", size=20, color="purple"),
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
            plot_bgcolor=plot_bgcolor,
            # plot_bgcolor='rgba(0, 0, 0, 0)',   # Transparent plot area
            showlegend=True,
        )

    def _plot_initial_statistics(self):
        # Create a box plot for each layer's initial weights
        fig = go.Figure()
        for layer_name, weights in self._initial_weights.items():
            fig.add_trace(
                go.Box(
                    y=weights,
                    name=layer_name,
                    boxmean="sd",
                    showlegend=False,  # Hide legend for simplicity
                    boxpoints=False,  # Hide individual points (outliers)
                )
            )  # boxmean='sd' adds a mean and standard deviation to the box plot

        self._format_fig(
            fig=fig,
            title="Initial Weight Distribution (Box Plot) Per Layer",
            xaxis_title="Layer",
            yaxis_title="Weight Value",
        )
        return fig

    def _plot_costs(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self._losses, mode="lines", name="Loss"))
        self._format_fig(
            fig=fig,
            title="Training Loss (Cost) Over Time",
            xaxis_title="Training Steps",
            yaxis_title="Loss",
        )
        return fig

    def _plot_grad_norms(self):
        # Create a plotly figure for the gradient norms
        fig = go.Figure()
        for layer_name, norm in self._gradient_norms.items():
            fig.add_trace(go.Scatter(y=norm, mode="lines", name=layer_name))
        self._format_fig(
            fig=fig,
            title="Gradient Norms Over Time",
            xaxis_title="Training Step Per Minibatch",
            yaxis_title="Gradient Norm (L2)",
        )
        return fig

    def _plot_accuracy(self):
        # Runs validation set in the model, then calculate accuracy
        y_test_labels = np.argmax(self._y_test, axis=-1)
        y_hat = (
            self._predict_func(X=torch.from_numpy(self._X_test), use_argmax=True)
            .detach()
            .numpy()
        )
        accuracy = np.sum(np.isclose(y_test_labels, y_hat)) / float(len(y_hat))

        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines"))  # Example line
        fig.add_annotation(
            text=f"{accuracy*100}%",
            x=0.5,  # X coordinate
            y=0.5,  # Y coordinate
            showarrow=False,
            font=dict(size=24, color="red"),
        )
        self._format_fig(
            fig=fig,
            title="Test Set Accuracy",
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor="BlueViolet ",
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(range=[0, 1], visible=False),
            showlegend=False,
        )
        return fig

    def plot_summary(self):
        app = dash.Dash(__name__)
        divs = []

        def get_div(id, func):
            return html.Div(
                children=[dcc.Graph(id=id, figure=func())],
                style={"width": "48%", "display": "inline-block"},
            )

        for i in range(0, len(self._plot_func_ids), 2):
            # Juxtapose the two plots side by side
            children = [
                get_div(func=self._plot_func_ids[i][0], id=self._plot_func_ids[i][1])
            ]
            if i + 1 < len(self._plot_func_ids):
                children.append(
                    get_div(
                        func=self._plot_func_ids[i + 1][0],
                        id=self._plot_func_ids[i + 1][1],
                    )
                )
            divs.append(html.Div(children=children))

        # Layout of the app
        app.layout = html.Div(
            children=[
                html.H1(
                    children="Training Summary",
                    style={
                        "font-family": "Arial",
                        "font-size": "36px",
                        "color": "black",
                        "text-align": "center",
                    },
                ),
            ]
            + divs,
            style={
                "background-color": "turquoise",  # Set the entire page background color to turquoise
                "min-height": "100vh",  # Ensure the background covers the full viewport height
            },
        )

        app.run_server(debug=True)


class FCN2DDebugger(FCNDebugger):
    def __init__(
        self, model: Module, config: FCNDebuggerConfig, X_test, y_test, predict_func
    ):
        super().__init__(model, config, X_test, y_test, predict_func=predict_func)
        self._plot_func_ids.append((self._plot_decision_boundary, "Decision Boundary"))

    def _plot_decision_boundary(self, resolution=0.02):
        """Decision Boundary Plot only for 2D data visualization

        Args:
            X (_type_): 2D numpy / torch tensor
            y (_type_): 1D numpy array
            resolution (float, optional): resolution for meshgrid
        """
        # TODO: THIS COULD BE LEAD TO A BUG: This is for softmax visualization. It's easier to work on labels
        # y here are labels
        y_test_labels = np.argmax(self._y_test, axis=-1)
        colors = ["red", "blue", "lightgreen", "gray", "cyan"]
        cmap = [colors[i] for i in np.unique(y_test_labels).astype(int)]
        # Create a meshgrid over the input space
        x1_min, x1_max = self._X_test[:, 0].min() - 1, self._X_test[:, 0].max() + 1
        x2_min, x2_max = self._X_test[:, 1].min() - 1, self._X_test[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
        )
        # Predict over the meshgrid
        Z = self._predict_func(
            X=torch.from_numpy(
                np.array([xx1.ravel(), xx2.ravel()]).astype(np.float32).T
            ),
            use_argmax=True,
        )
        Z = Z.reshape(xx1.shape).detach().numpy()
        # Create a contour plot for the decision boundary
        contour = go.Contour(
            x=np.arange(x1_min, x1_max, resolution),
            y=np.arange(x2_min, x2_max, resolution),
            z=Z,
            colorscale=cmap,
            opacity=0.3,
            showscale=False,
        )
        # Create scatter plots for the data points
        scatter_data = []
        for idx, cl in enumerate(np.unique(y_test_labels)):
            scatter = go.Scatter(
                x=self._X_test[np.reshape(y_test_labels == cl, (-1,)), 0],
                y=self._X_test[np.reshape(y_test_labels == cl, (-1,)), 1],
                mode="markers",
                name=f"Class {cl}",
                marker=dict(
                    color=colors[idx], size=8, line=dict(width=2, color="black")
                ),
            )
            scatter_data.append(scatter)
        fig = go.Figure(data=[contour] + scatter_data)
        self._format_fig(
            fig=fig,
            title="Decision Boundary",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
        )
        return fig


class CNNDebugger(FCNDebugger):
    def _get_feature_maps(self):
        # Grabs a feature map and stores it
        pass

    def _visualize_feature_maps(self):
        pass


if __name__ == "__main__":
    pass
    # sample usage:
    model = 1
    nn_config = FCNDebuggerConfig()
    nn_debugger = FCNDebugger(model=model, config=nn_config)
