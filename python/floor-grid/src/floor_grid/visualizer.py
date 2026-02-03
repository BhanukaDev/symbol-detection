import cv2
import numpy as np
from typing import List, Tuple, Set
from .models import Grid, Room, Door


class FloorPlanVisualizer:
    def __init__(self, grid: Grid, rooms: List[Room], doors: List[Door], cell_size=20):
        """
        grid: Grid object
        rooms: List of Room objects
        doors: List of Door objects
        cell_size: Size of each cell in pixels
        """
        self.grid = grid
        self.rooms = rooms
        self.doors = doors
        self.cell_size = cell_size
        self.wall_thickness = 8

        # Calculate image dimensions
        self.img_width = grid.cols * cell_size
        self.img_height = grid.rows * cell_size

        self.all_room_cells = set()
        for room in self.rooms:
            self.all_room_cells |= set(room.cells)

        # Colors (BGR format)
        self.bg_color = (240, 240, 240)  # Light gray background
        self.wall_color = (80, 80, 80)  # Dark gray walls
        self.wall_pattern_color = (100, 100, 100)  # Slightly lighter for pattern
        self.door_color = (139, 69, 19)  # Brown doors
        self.room_colors = self._generate_room_colors()

    def _generate_room_colors(self):
        """Generate distinct pastel colors for each room."""
        # Predefined pastel colors (BGR format)
        base_colors = [
            (255, 255, 255),  # White
        ]

        colors = []
        for i in range(len(self.rooms)):
            colors.append(base_colors[i % len(base_colors)])
        return colors

    def create_floor_plan(self):
        """
        Create the complete floor plan image.
        """
        # Create blank image with background color
        img = np.full(
            (self.img_height, self.img_width, 3), self.bg_color, dtype=np.uint8
        )

        # Draw rooms (filled areas)
        img = self._draw_rooms(img)

        # Draw walls
        img = self._draw_walls(img)

        # Draw doors (this will clear walls where doors are)
        img = self._draw_doors(img)

        return img

    def _draw_rooms(self, img):
        """
        Draw room backgrounds with different colors.
        """
        for idx, room in enumerate(self.rooms):
            color = self.room_colors[idx]
            for r, c in room.cells:
                x = c * self.cell_size
                y = r * self.cell_size
                cv2.rectangle(
                    img, (x, y), (x + self.cell_size, y + self.cell_size), color, -1
                )
        return img

    def _draw_walls(self, img):
        """
        Draw all walls with proper thickness and pattern.
        """
        # Collect all wall segments
        wall_segments = {"horizontal": [], "vertical": []}

        for room in self.rooms:
            for r, c in room.cells:
                x = c * self.cell_size
                y = r * self.cell_size

                # Check each edge and draw wall if needed
                # Top edge
                if (r - 1, c) not in room.cells:
                    wall_segments["horizontal"].append(
                        {
                            "x1": x,
                            "y1": y,
                            "x2": x + self.cell_size,
                            "y2": y,
                            "r": r,
                            "c": c,
                            "side": "top",
                        }
                    )

                # Bottom edge
                if (r + 1, c) not in room.cells:
                    wall_segments["horizontal"].append(
                        {
                            "x1": x,
                            "y1": y + self.cell_size,
                            "x2": x + self.cell_size,
                            "y2": y + self.cell_size,
                            "r": r,
                            "c": c,
                            "side": "bottom",
                        }
                    )

                # Left edge
                if (r, c - 1) not in room.cells:
                    wall_segments["vertical"].append(
                        {
                            "x1": x,
                            "y1": y,
                            "x2": x,
                            "y2": y + self.cell_size,
                            "r": r,
                            "c": c,
                            "side": "left",
                        }
                    )

                # Right edge
                if (r, c + 1) not in room.cells:
                    wall_segments["vertical"].append(
                        {
                            "x1": x + self.cell_size,
                            "y1": y,
                            "x2": x + self.cell_size,
                            "y2": y + self.cell_size,
                            "r": r,
                            "c": c,
                            "side": "right",
                        }
                    )

        # Draw horizontal walls
        for wall in wall_segments["horizontal"]:
            x1, y1, x2, y2 = wall["x1"], wall["y1"], wall["x2"], wall["y2"]

            # Draw thick wall line
            cv2.line(img, (x1, y1), (x2, y2), self.wall_color, self.wall_thickness)

            # Add diagonal pattern
            pattern_rect = {
                "x": x1,
                "y": y1 - self.wall_thickness // 2,
                "width": x2 - x1,
                "height": self.wall_thickness,
            }
            img = self._draw_wall_pattern(img, pattern_rect)

        # Draw vertical walls
        for wall in wall_segments["vertical"]:
            x1, y1, x2, y2 = wall["x1"], wall["y1"], wall["x2"], wall["y2"]

            # Draw thick wall line
            cv2.line(img, (x1, y1), (x2, y2), self.wall_color, self.wall_thickness)

            # Add diagonal pattern
            pattern_rect = {
                "x": x1 - self.wall_thickness // 2,
                "y": y1,
                "width": self.wall_thickness,
                "height": y2 - y1,
            }
            img = self._draw_wall_pattern(img, pattern_rect)

        return img

    def _draw_wall_pattern(self, img, rect):
        """
        Draw diagonal line pattern in a rectangle.
        """
        x = int(rect["x"])
        y = int(rect["y"])
        width = int(rect["width"])
        height = int(rect["height"])

        line_spacing = 4

        # Draw diagonal lines
        for i in range(-height, width + height, line_spacing):
            pt1_x = x + i
            pt1_y = y
            pt2_x = x
            pt2_y = y + i

            # Clip to rectangle bounds
            if pt1_x > x + width:
                offset = pt1_x - (x + width)
                pt1_x = x + width
                pt1_y = y + offset

            if pt2_y > y + height:
                offset = pt2_y - (y + height)
                pt2_y = y + height
                pt2_x = x + offset

            # Make sure points are within bounds
            if pt1_x < x:
                pt1_x = x
            if pt2_x < x:
                pt2_x = x
            if pt1_y < y:
                pt1_y = y
            if pt2_y < y:
                pt2_y = y

            if pt1_y <= y + height and pt2_x <= x + width and pt1_x >= x and pt2_y >= y:
                cv2.line(
                    img,
                    (int(pt1_x), int(pt1_y)),
                    (int(pt2_x), int(pt2_y)),
                    self.wall_pattern_color,
                    1,
                )

        return img

    def _draw_doors(self, img):
        """
        Draw all doors with arc shape.
        """
        for door in self.doors:
            img = self.draw_door(img, door)
        return img

    def draw_door(self, img, door):
        """
        Draw a single door with arc shape showing door swing.
        Correctly detects the wall side based on room ownership of door cells.
        """
        if not door.cells or len(door.cells) < 2:
            return img

        cells = door.cells

        # Find the room that owns these door cells
        owner_room = None
        # Try to find room in door.room1 or door.room2
        for r_cand in [door.room1, door.room2]:
            if r_cand and cells[0] in r_cand.cells:
                owner_room = r_cand
                break

        # If not found (e.g. entrance door or mismatch), search all rooms
        if not owner_room:
            for r_cand in self.rooms:
                if cells[0] in r_cand.cells:
                    owner_room = r_cand
                    break

        # Fallback set for membership testing if room not found
        owner_cells = set(owner_room.cells) if owner_room else self.all_room_cells

        t = self.wall_thickness
        half = t // 2
        clear_pad = 1  # small, not 2+ pixels

        # If same row => door spans along X, wall is horizontal (y constant)
        if cells[0][0] == cells[1][0]:
            r = cells[0][0]
            c_min = min(c for _, c in cells)
            c_max = max(c for _, c in cells)

            x1 = c_min * self.cell_size
            x2 = (c_max + 1) * self.cell_size

            # Check neighbors relative to owner_room to find "outside"
            # Since cells are IN the room, the wall is on the side where neighbors are NOT in the room.

            # Check top neighbor (r-1)
            top_is_outside = any(
                (r - 1, c) not in owner_cells for c in range(c_min, c_max + 1)
            )

            # Check bottom neighbor (r+1)
            bottom_is_outside = any(
                (r + 1, c) not in owner_cells for c in range(c_min, c_max + 1)
            )

            if top_is_outside:
                wall_y = r * self.cell_size  # top edge of row r cells
                swing_sign = +1  # swing into below => +y (inside room)
            else:
                wall_y = (r + 1) * self.cell_size  # bottom edge
                swing_sign = -1  # swing into above => -y (inside room)

            # Place hinge on the wall *face* (not centerline)
            hinge_y = wall_y + swing_sign * half

            # Clear only the wall thickness region (no big padding)
            cv2.rectangle(
                img,
                (x1, wall_y - half - clear_pad),
                (x2, wall_y + half + clear_pad),
                self.bg_color,
                -1,
            )

            door_width = x2 - x1
            radius = door_width

            center = (x1, hinge_y)

            # Arc angles depend on swing direction
            if swing_sign < 0:  # opens upward
                cv2.ellipse(
                    img, center, (radius, radius), 0, 270, 360, self.door_color, 2
                )
                angle = 315
            else:  # opens downward
                cv2.ellipse(img, center, (radius, radius), 0, 0, 90, self.door_color, 2)
                angle = 45

            end_x = center[0] + int(radius * np.cos(np.radians(angle)))
            end_y = center[1] + int(radius * np.sin(np.radians(angle)))
            cv2.line(img, center, (end_x, end_y), self.door_color, 2)

        # If same column => door spans along Y, wall is vertical (x constant)
        else:
            c = cells[0][1]
            r_min = min(r for r, _ in cells)
            r_max = max(r for r, _ in cells)

            y1 = r_min * self.cell_size
            y2 = (r_max + 1) * self.cell_size

            # Check neighbors relative to owner_room

            # Check left neighbor (c-1)
            left_is_outside = any(
                (r, c - 1) not in owner_cells for r in range(r_min, r_max + 1)
            )

            # Check right neighbor (c+1)
            right_is_outside = any(
                (r, c + 1) not in owner_cells for r in range(r_min, r_max + 1)
            )

            if left_is_outside:
                wall_x = c * self.cell_size  # left edge of col c cells
                swing_sign = +1  # swing into right => +x (inside room)
            else:
                wall_x = (c + 1) * self.cell_size  # right edge
                swing_sign = -1  # swing into left => -x (inside room)

            hinge_x = wall_x + swing_sign * half

            # Clear only the wall thickness region
            cv2.rectangle(
                img,
                (wall_x - half - clear_pad, y1),
                (wall_x + half + clear_pad, y2),
                self.bg_color,
                -1,
            )

            door_height = y2 - y1
            radius = door_height

            center = (hinge_x, y1)

            if swing_sign < 0:  # opens left
                cv2.ellipse(
                    img, center, (radius, radius), 90, 270, 360, self.door_color, 2
                )
                angle = 135
            else:  # opens right
                cv2.ellipse(
                    img, center, (radius, radius), 90, 0, 90, self.door_color, 2
                )
                angle = 45

            end_x = center[0] + int(radius * np.cos(np.radians(angle)))
            end_y = center[1] + int(radius * np.sin(np.radians(angle)))
            cv2.line(img, center, (end_x, end_y), self.door_color, 2)

        # Mark entrance (unchanged)
        if door.is_entrance and cells:
            r, c = cells[len(cells) // 2]
            x = c * self.cell_size + self.cell_size // 2
            y = r * self.cell_size + self.cell_size // 2
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(
                img,
                "E",
                (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return img

    def add_room_labels(self, img):
        """
        Add room labels/numbers to the image.
        """
        for idx, room in enumerate(self.rooms):
            if room.cells:
                avg_r = sum(r for r, c in room.cells) / len(room.cells)
                avg_c = sum(c for r, c in room.cells) / len(room.cells)

                x = int(avg_c * self.cell_size)
                y = int(avg_r * self.cell_size)

                label = f"R{idx}"
                cv2.putText(
                    img,
                    label,
                    (x - 15, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

        return img

    def save_image(self, img, filename="floor_plan.png"):
        """Save the floor plan image to file."""
        cv2.imwrite(filename, img)
        print(f"Floor plan saved to {filename}")

    def show_image(self, img):
        """Display the floor plan image."""
        cv2.imshow("Floor Plan", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
