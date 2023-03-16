"""Util to plot BEV."""
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import PIL


from .bev_utils import BEVSettings, create_pointcloud_input


class BEVBoxAnimation:
    """Write objects on top of a discrete bird's eye view tensor."""

    def __init__(self):
        """Construct instance."""
        super().__init__()
        self._settings = BEVSettings()
        # Exclude dark gray color
        self._color = px.colors.qualitative.Dark24[:5] + px.colors.qualitative.Dark24[6:]

    def __call__(
        self, bev: List[np.ndarray], objects: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], save_path: str
    ) -> go.Figure:
        
        frames = []
        """Go through all frames in sequence"""
        for i in range(len(bev)):
            """Go through all objects for each batch, and write them on top the BEV."""
            input_ = create_pointcloud_input(bev[i], self._settings)
            classes, positions, dimensions, rotations = objects[i]
            n_objects = np.shape(positions)[0]

            # Setup figure
            fig = self._setup_figure()

            # Plot the reduced BEV
            input_ = self._create_od_vis_background(input_)
            fig.add_trace(go.Image(z=input_))
            print('n_objects', n_objects)

            # Add object rectangles, if any
            if n_objects > 0:
                obs = []
                # Map classes
                classes = self._settings.encode_classes(classes)
                # Scale dimensions and positions
                dimensions = dimensions[:, :2] / self._settings.grid_cell_size
                positions = (positions[:, :2] - self._settings.grid_min) / self._settings.grid_cell_size
                for obj_idx in range(n_objects):
                    self._add_object(
                        fig,
                        classes[obj_idx],
                        positions[obj_idx],
                        dimensions[obj_idx],
                        rotations[obj_idx],
                    )

            frames += [PIL.Image.open(io.BytesIO(fig.to_image(format="png")))]
        print(frames)
        self._save_animation(frames, save_path)
        return fig

    def _setup_figure(self) -> go.Figure:
        limits = (self._settings.grid_max - self._settings.grid_min) / self._settings.grid_cell_size
        # Setup a figure with correct scaling and ticks.
        xtickvals, xticktext, ytickvals, yticktext = self._calculate_ticks(limits)
        layout = go.Layout(
            width=1000,
            height=1000,
            xaxis=dict(
                range=[0, int(limits[0])],
                title="Lateral (meters)",
                tickvals=xtickvals,
                ticktext=xticktext,
            ),
            yaxis=dict(
                range=[0, int(limits[1])],
                title="Longitudinal (meters)",
                tickvals=ytickvals,
                ticktext=yticktext,
                scaleanchor="x",
            ),
        )
        fig = go.Figure(layout=layout)
        return fig
    
    def _setup_animation(self, fig) -> go.Figure:
        limits = (self._settings.grid_max - self._settings.grid_min) / self._settings.grid_cell_size
        xtickvals, xticktext, ytickvals, yticktext = self._calculate_ticks(limits)

        def frame_args(duration):
            return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

        sliders = [
                    {
                        "pad": {"b": 10, "t": 60},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [[f.name], frame_args(0)],
                                "label": str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig.frames)
                        ],
                    }
                ]

        fig.update_layout(
            title='Sequence animation',
            width=1000,
            height=1000,
            xaxis=dict(
                range=[0, int(limits[0])],
                title="Lateral (meters)",
                tickvals=xtickvals,
                ticktext=xticktext,
            ),
            yaxis=dict(
                range=[0, int(limits[1])],
                title="Longitudinal (meters)",
                tickvals=ytickvals,
                ticktext=yticktext,
                scaleanchor="x",
            ),
            #scene=dict(
            #            zaxis=dict(range=[-0.1, 6.8], autorange=False),
            #            aspectratio=dict(x=1, y=1, z=1),
            #            ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
        )
        return fig

    def _save_animation(self, frames, save_path):
        #gif.save(frames, save_path, duration=100)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=500,
            loop=0,
        )
    def _calculate_ticks(
        self, limits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, ax = plt.subplots(1, 1)
        ax.set_aspect("equal")
        ax.set_xlim(0, int(limits[0]))
        ax.set_ylim(0, int(limits[1]))
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_ticks_text = [
            tick * self._settings.grid_cell_size + self._settings.grid_min[0] for tick in x_ticks
        ]
        y_ticks_text = [
            tick * self._settings.grid_cell_size + self._settings.grid_min[1] for tick in y_ticks
        ]
        plt.close()
        return x_ticks, x_ticks_text, y_ticks, y_ticks_text

    def _add_object(
        self,
        fig: go.Figure,
        class_ind: int,
        position: np.ndarray,
        dimension: np.ndarray,
        rotation: np.ndarray,
    ):
        # pylint: disable=too-many-arguments
        # Extract object information
        coord1 = (-1) * dimension / 2
        coord3 = dimension / 2
        coord2 = [coord1[0], coord3[1]]
        coord4 = [coord3[0], coord1[1]]

        all_coords = np.vstack([coord1, coord2, coord3, coord4])
        all_coords = np.hstack([all_coords, np.zeros((4, 1))])

        all_coords = [rotation.rotate(point)[:2] + position for point in all_coords]
        all_coords = np.vstack([all_coords, all_coords[0]])

        # Add rotated bounding box
        fig.add_scatter(
            x=all_coords[:, 0],
            y=all_coords[:, 1],
            line=dict(color=self._color[class_ind], width=self._settings.grid_cell_size * 30),
            marker=dict(size=self._settings.grid_cell_size * 30),
            legendgroup=self._settings.get_class_name(class_ind),
            name=self._settings.get_class_name(class_ind),
            showlegend=False,
        )
        '''
        fig.add_trace(go.Scatter(
            x=all_coords[:, 0],
            y=all_coords[:, 1],
            line=dict(color=self._color[class_ind], width=self._settings.grid_cell_size * 30),
            marker=dict(size=self._settings.grid_cell_size * 30),
            legendgroup=self._settings.get_class_name(class_ind),
            name=self._settings.get_class_name(class_ind),
            showlegend=False,
        ))
        '''
        
        # Add arrow in the direction of the object
        arrow_length = int(np.cast["float32"](dimension[0]) * 0.8)
        end_point = np.array([arrow_length, 0, 0])
        end_point = rotation.rotate(end_point)[:2] + position
        fig.add_annotation(
            x=end_point[0],
            y=end_point[1],
            xref="x",
            yref="y",
            text="",
            showarrow=True,
            axref="x",
            ayref="y",
            ax=position[0],
            ay=position[1],
            arrowhead=3,
            arrowwidth=self._settings.grid_cell_size * 20,
            arrowcolor="#247BA0",
        )
        #fig.update_layout(showlegend=False)


    def _activate_legend(self, fig):
        """Activate legend with all classes."""
        for idx, _ in enumerate(self._settings.classes):
            fig.add_scatter(
                x=[0],
                y=[0],
                legendgroup=self._settings.get_class_name(idx),
                name=self._settings.get_class_name(idx),
                showlegend=True,
                marker=dict(color=self._color[idx], size=0.01),
            )

    @staticmethod
    def _create_od_vis_background(input_array: np.ndarray) -> np.ndarray:
        """Create a gray occupancy grid as background to visualize over."""
        occupancy = np.maximum.reduce(
            np.cast["float32"](np.abs(input_array) > 0.0), axis=0, keepdims=True
        )
        vis_bg = np.transpose(np.repeat(occupancy * 77, 3, axis=0), [2, 1, 0])
        return vis_bg
