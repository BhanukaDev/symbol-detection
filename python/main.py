import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from typing import Optional


class Cell:
    def __init__(self, id, posx=None, posy=None, is_building_cell=True):
        self.id = id
        self.posx = posx
        self.posy = posy
        self.is_building_cell = is_building_cell


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = self._create_grid()

    def _create_grid(self):
        # Create a 2D list of Cell objects
        grid = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                cell_id = (r, c)  # Using (row, col) as id for now
                row_cells.append(Cell(cell_id, posx=c, posy=r))  # Added posx and posy
            grid.append(row_cells)
        return grid

    def is_valid_coord(self, r, c):
        # Helper to check if coordinates are within grid bounds
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_cell(self, r, c) -> Optional[Cell]:
        # Get a cell by its row and column
        if self.is_valid_coord(r, c):
            return self.grid[r][c]
        return None  # Or raise an error for invalid coordinates

    def find_neighbors(self, r, c):
        # Returns a list of valid neighbor cell objects (up, down, left, right)
        neighbors = []
        # Define possible movements (dr, dc)
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for dr, dc in movements:
            new_r, new_c = r + dr, c + dc
            if self.is_valid_coord(new_r, new_c):
                neighbors.append(self.get_cell(new_r, new_c))
        return neighbors

    def is_same_row(self, cell1_coords, cell2_coords):
        # Check if two cells are in the same row
        return cell1_coords[0] == cell2_coords[0]

    def is_same_column(self, cell1_coords, cell2_coords):
        # Check if two cells are in the same column
        return cell1_coords[1] == cell2_coords[1]

    def get_cells_in_direction(self, r, c, direction, distance):
        """
        Returns a list of cells in a specified direction from a starting cell
        up to a given distance.

        Args:
            r (int): Row of the starting cell.
            c (int): Column of the starting cell.
            direction (str): 'up', 'down', 'left', or 'right'.
            distance (int): Number of cells to retrieve in the specified direction.

        Returns:
            list: A list of Cell objects.
        """
        result_cells = []
        dr, dc = 0, 0

        if direction == "up":
            dr = -1
        elif direction == "down":
            dr = 1
        elif direction == "left":
            dc = -1
        elif direction == "right":
            dc = 1
        else:
            raise ValueError("Direction must be 'up', 'down', 'left', or 'right'")

        for i in range(
            1, distance + 1
        ):  # Start from 1 to get cells *from* the current cell
            new_r, new_c = r + dr * i, c + dc * i
            if self.is_valid_coord(new_r, new_c):
                result_cells.append(self.get_cell(new_r, new_c))
            else:
                break  # Hit grid boundary
        return result_cells

    def display_grid_ids(self):
        # Simple helper to print the IDs of the cells in the grid
        for r in range(self.rows):
            row_ids = [self.grid[r][c].id for c in range(self.cols)]
            print(row_ids)


base_grid = Grid(28, 32)
door_size = 2  # 2 grids wide
min_room_width = 5  # a room's width or length can't be less than this
min_building_area_ratio = (
    0.6  # from all the cells of grid, this amount of cells should be inside building
)


import random
from collections import deque


class BuildingShapeGenerator:
    def __init__(
        self, grid, door_size=2, min_room_width=5, min_building_area_ratio=0.6
    ):
        self.grid = grid
        self.door_size = door_size
        self.min_room_width = min_room_width
        self.min_building_area_ratio = min_building_area_ratio
        self.total_cells = grid.rows * grid.cols
        self.target_building_cells = int(self.total_cells * min_building_area_ratio)

    def generate_building_shape(self):
        """
        Generate a connected building shape with smooth, rectangular edges.
        """
        # Start with a rectangular base in the center
        center_r = self.grid.rows // 2
        center_c = self.grid.cols // 2

        # Initialize all cells as non-building
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                self.grid.grid[r][c].is_building_cell = False

        # Create initial rectangle
        initial_width = self.min_room_width * 2
        initial_height = self.min_room_width * 2

        start_r = max(0, center_r - initial_height // 2)
        end_r = min(self.grid.rows, center_r + initial_height // 2)
        start_c = max(0, center_c - initial_width // 2)
        end_c = min(self.grid.cols, center_c + initial_width // 2)

        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                self.grid.grid[r][c].is_building_cell = True

        # Grow the building by adding rectangular chunks
        current_cells = self._count_building_cells()

        while current_cells < self.target_building_cells:
            # Try to add a rectangular extension
            if not self._add_rectangular_extension():
                break
            current_cells = self._count_building_cells()

        # Final cleanup to ensure smooth edges
        self._smooth_edges()

        return self.grid

    def _add_rectangular_extension(self):
        """
        Add a rectangular extension to the building.
        Extensions are at least min_room_width in size.
        """
        # Find all edge cells (building cells with at least one non-building neighbor)
        edge_cells = []
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    neighbors = self.grid.find_neighbors(r, c)
                    if any(not n.is_building_cell for n in neighbors):
                        edge_cells.append((r, c))

        if not edge_cells:
            return False

        # Try random edge cells until we find one we can extend
        random.shuffle(edge_cells)

        for r, c in edge_cells:
            # Try each direction
            directions = ["up", "down", "left", "right"]
            random.shuffle(directions)

            for direction in directions:
                if self._try_extend_direction(r, c, direction):
                    return True

        return False

    def _try_extend_direction(self, r, c, direction):
        """
        Try to extend the building in a given direction with a rectangular block.
        """
        # Determine the extension size (random between min_room_width and min_room_width * 2)
        extension_depth = random.randint(self.min_room_width, self.min_room_width * 2)
        extension_width = random.randint(self.min_room_width, self.min_room_width * 3)

        cells_to_add = []

        if direction == "up" or direction == "down":
            # Vertical extension
            dr = -1 if direction == "up" else 1

            # Center the extension around the current column
            start_c = max(0, c - extension_width // 2)
            end_c = min(self.grid.cols, start_c + extension_width)

            # Check if we can extend
            for depth in range(1, extension_depth + 1):
                new_r = r + dr * depth
                if not self.grid.is_valid_coord(new_r, 0):
                    break

                for check_c in range(start_c, end_c):
                    if not self.grid.is_valid_coord(new_r, check_c):
                        return False
                    if self.grid.grid[new_r][check_c].is_building_cell:
                        continue
                    cells_to_add.append((new_r, check_c))

        elif direction == "left" or direction == "right":
            # Horizontal extension
            dc = -1 if direction == "left" else 1

            # Center the extension around the current row
            start_r = max(0, r - extension_width // 2)
            end_r = min(self.grid.rows, start_r + extension_width)

            # Check if we can extend
            for depth in range(1, extension_depth + 1):
                new_c = c + dc * depth
                if not self.grid.is_valid_coord(0, new_c):
                    break

                for check_r in range(start_r, end_r):
                    if not self.grid.is_valid_coord(check_r, new_c):
                        return False
                    if self.grid.grid[check_r][new_c].is_building_cell:
                        continue
                    cells_to_add.append((check_r, new_c))

        # If we have valid cells to add and they form a reasonable extension
        if len(cells_to_add) >= self.min_room_width:
            for cell_r, cell_c in cells_to_add:
                self.grid.grid[cell_r][cell_c].is_building_cell = True
            return True

        return False

    def _smooth_edges(self):
        """
        Remove jagged edges that are smaller than min_room_width.
        """
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Check for vertical protrusions
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.grid[r][c].is_building_cell:
                        # Check if this is a vertical protrusion
                        if self._is_vertical_protrusion(r, c):
                            self.grid.grid[r][c].is_building_cell = False
                            changed = True

            # Check for horizontal protrusions
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.grid[r][c].is_building_cell:
                        # Check if this is a horizontal protrusion
                        if self._is_horizontal_protrusion(r, c):
                            self.grid.grid[r][c].is_building_cell = False
                            changed = True

    def _is_vertical_protrusion(self, r, c):
        """
        Check if this cell is part of a vertical protrusion smaller than min_room_width.
        """
        # Count continuous building cells to the left and right
        left_count = 0
        for check_c in range(c - 1, -1, -1):
            if self.grid.grid[r][check_c].is_building_cell:
                left_count += 1
            else:
                break

        right_count = 0
        for check_c in range(c + 1, self.grid.cols):
            if self.grid.grid[r][check_c].is_building_cell:
                right_count += 1
            else:
                break

        total_width = left_count + 1 + right_count

        # Check if this is an edge cell
        up_is_building = r > 0 and self.grid.grid[r - 1][c].is_building_cell
        down_is_building = (
            r < self.grid.rows - 1 and self.grid.grid[r + 1][c].is_building_cell
        )

        # If it's an edge and the width is less than min_room_width, it's a protrusion
        if (
            not up_is_building or not down_is_building
        ) and total_width < self.min_room_width:
            return True

        return False

    def _is_horizontal_protrusion(self, r, c):
        """
        Check if this cell is part of a horizontal protrusion smaller than min_room_width.
        """
        # Count continuous building cells up and down
        up_count = 0
        for check_r in range(r - 1, -1, -1):
            if self.grid.grid[check_r][c].is_building_cell:
                up_count += 1
            else:
                break

        down_count = 0
        for check_r in range(r + 1, self.grid.rows):
            if self.grid.grid[check_r][c].is_building_cell:
                down_count += 1
            else:
                break

        total_height = up_count + 1 + down_count

        # Check if this is an edge cell
        left_is_building = c > 0 and self.grid.grid[r][c - 1].is_building_cell
        right_is_building = (
            c < self.grid.cols - 1 and self.grid.grid[r][c + 1].is_building_cell
        )

        # If it's an edge and the height is less than min_room_width, it's a protrusion
        if (
            not left_is_building or not right_is_building
        ) and total_height < self.min_room_width:
            return True

        return False

    def _count_building_cells(self):
        """Count the number of building cells."""
        return sum(
            1
            for r in range(self.grid.rows)
            for c in range(self.grid.cols)
            if self.grid.grid[r][c].is_building_cell
        )

    def is_connected(self):
        """
        Check if all building cells form a connected shape (no floating islands).
        Uses BFS to verify connectivity.
        """
        # Find first building cell
        start_cell = None
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    start_cell = (r, c)
                    break
            if start_cell:
                break

        if not start_cell:
            return False

        # BFS to find all connected building cells
        visited = set()
        queue = deque([start_cell])
        visited.add(start_cell)

        while queue:
            r, c = queue.popleft()
            neighbors = self.grid.find_neighbors(r, c)

            for neighbor in neighbors:
                coord = (neighbor.posy, neighbor.posx)
                if neighbor.is_building_cell and coord not in visited:
                    visited.add(coord)
                    queue.append(coord)

        # Count total building cells
        total_building = self._count_building_cells()

        return len(visited) == total_building

    def display_building_shape(self):
        """
        Display the building shape in ASCII.
        ■ = building cell
        □ = non-building cell
        """
        for r in range(self.grid.rows):
            row_display = []
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    row_display.append("■")
                else:
                    row_display.append("□")
            print(" ".join(row_display))

        # Print statistics
        building_count = self._count_building_cells()
        actual_ratio = building_count / self.total_cells
        print(
            f"\nBuilding cells: {building_count}/{self.total_cells} ({actual_ratio:.2%})"
        )
        print(f"Target ratio: {self.min_building_area_ratio:.2%}")
        print(f"Connected: {self.is_connected()}")


import random
from typing import List, Tuple, Set


class Room:
    def __init__(self, cell_coords):
        """
        cell_coords: list of (r, c) tuples representing cells in this room
        """
        self.cells = set(cell_coords)  # Set of (r, c) coordinates
        self.walls = []  # Will store wall segments later

    def get_bounds(self):
        """Get the bounding box of this room."""
        if not self.cells:
            return None

        min_r = min(r for r, c in self.cells)
        max_r = max(r for r, c in self.cells)
        min_c = min(c for r, c in self.cells)
        max_c = max(c for r, c in self.cells)

        return (min_r, max_r, min_c, max_c)

    def get_width(self):
        """Get the width of the room's bounding box."""
        bounds = self.get_bounds()
        if not bounds:
            return 0
        return bounds[3] - bounds[2] + 1

    def get_height(self):
        """Get the height of the room's bounding box."""
        bounds = self.get_bounds()
        if not bounds:
            return 0
        return bounds[1] - bounds[0] + 1

    def get_area(self):
        """Get the number of cells in this room."""
        return len(self.cells)

    def is_rectangular(self):
        """Check if the room is a perfect rectangle."""
        return self.get_area() == self.get_width() * self.get_height()


class RoomDivider:
    def __init__(self, grid, min_room_width=5):
        self.grid = grid
        self.min_room_width = min_room_width  # This is actually min_wall_width
        self.rooms = []

    def divide_into_rooms(self):
        """
        Main method to divide the building into rooms.
        """
        # Get all building cells
        building_cells = []
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    building_cells.append((r, c))

        if not building_cells:
            return []

        # Start with the entire building as one region
        initial_region = set(building_cells)

        # Recursively divide the region into rooms
        self._divide_region(initial_region)

        # Smooth each room to remove small protrusions
        self._smooth_all_rooms()

        # Optionally create some L-shaped rooms by merging adjacent rooms
        self._create_l_shapes()

        # Final smoothing after merging
        self._smooth_all_rooms()

        return self.rooms

    def _divide_region(self, region_cells, depth=0, max_depth=10):
        """
        Recursively divide a region into smaller rooms.
        """
        if depth >= max_depth:
            # Max depth reached, create a room
            if len(region_cells) >= self.min_room_width * self.min_room_width:
                self.rooms.append(Room(region_cells))
            return

        # Get bounding box of the region
        min_r = min(r for r, c in region_cells)
        max_r = max(r for r, c in region_cells)
        min_c = min(c for r, c in region_cells)
        max_c = max(c for r, c in region_cells)

        width = max_c - min_c + 1
        height = max_r - min_r + 1

        # Check if region is too small to divide
        if width < self.min_room_width * 2 and height < self.min_room_width * 2:
            self.rooms.append(Room(region_cells))
            return

        # Decide whether to split horizontally or vertically
        can_split_horizontal = height >= self.min_room_width * 2
        can_split_vertical = width >= self.min_room_width * 2

        if not can_split_horizontal and not can_split_vertical:
            self.rooms.append(Room(region_cells))
            return

        # Randomly choose split direction (weighted by available space)
        if can_split_horizontal and can_split_vertical:
            split_horizontal = random.choice([True, False])
        elif can_split_horizontal:
            split_horizontal = True
        else:
            split_horizontal = False

        # Perform the split
        if split_horizontal:
            split_success = self._split_horizontal(region_cells, min_r, max_r, depth)
        else:
            split_success = self._split_vertical(region_cells, min_c, max_c, depth)

        # If split failed, create a room from the entire region
        if not split_success:
            self.rooms.append(Room(region_cells))

    def _split_horizontal(self, region_cells, min_r, max_r, depth):
        """
        Split a region horizontally (along a row).
        """
        # Choose a random split position
        possible_splits = list(
            range(min_r + self.min_room_width, max_r - self.min_room_width + 2)
        )

        if not possible_splits:
            return False

        random.shuffle(possible_splits)

        for split_row in possible_splits:
            # Divide cells into top and bottom regions
            top_region = set()
            bottom_region = set()

            for r, c in region_cells:
                if r < split_row:
                    top_region.add((r, c))
                else:
                    bottom_region.add((r, c))

            # Check if both regions are valid
            if self._is_region_connected(top_region) and self._is_region_connected(
                bottom_region
            ):
                # Recursively divide both regions
                self._divide_region(top_region, depth + 1)
                self._divide_region(bottom_region, depth + 1)
                return True

        return False

    def _split_vertical(self, region_cells, min_c, max_c, depth):
        """
        Split a region vertically (along a column).
        """
        # Choose a random split position
        possible_splits = list(
            range(min_c + self.min_room_width, max_c - self.min_room_width + 2)
        )

        if not possible_splits:
            return False

        random.shuffle(possible_splits)

        for split_col in possible_splits:
            # Divide cells into left and right regions
            left_region = set()
            right_region = set()

            for r, c in region_cells:
                if c < split_col:
                    left_region.add((r, c))
                else:
                    right_region.add((r, c))

            # Check if both regions are valid
            if self._is_region_connected(left_region) and self._is_region_connected(
                right_region
            ):
                # Recursively divide both regions
                self._divide_region(left_region, depth + 1)
                self._divide_region(right_region, depth + 1)
                return True

        return False

    def _smooth_all_rooms(self):
        """
        Smooth all rooms to remove small protrusions.
        """
        for room in self.rooms:
            self._smooth_room(room)

        # Remove any rooms that became too small after smoothing
        self.rooms = [
            room
            for room in self.rooms
            if room.get_area() >= self.min_room_width * self.min_room_width
        ]

    def _smooth_room(self, room):
        """
        Remove jagged edges from a room that are smaller than min_room_width.
        """
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            cells_to_remove = []

            # Check each cell in the room
            for r, c in room.cells:
                # Check if this cell is part of a small protrusion
                if self._is_vertical_protrusion_in_room(room, r, c):
                    cells_to_remove.append((r, c))
                    changed = True
                elif self._is_horizontal_protrusion_in_room(room, r, c):
                    cells_to_remove.append((r, c))
                    changed = True

            # Remove the cells
            for cell in cells_to_remove:
                room.cells.discard(cell)

            # Make sure room is still connected
            if not self._is_region_connected(room.cells):
                # If disconnected, we need to keep only the largest connected component
                room.cells = self._get_largest_connected_component(room.cells)

    def _is_vertical_protrusion_in_room(self, room, r, c):
        """
        Check if this cell is part of a vertical protrusion smaller than min_room_width within the room.
        """
        # Count continuous room cells to the left and right
        left_count = 0
        for check_c in range(c - 1, -1, -1):
            if (r, check_c) in room.cells:
                left_count += 1
            else:
                break

        right_count = 0
        for check_c in range(c + 1, self.grid.cols):
            if (r, check_c) in room.cells:
                right_count += 1
            else:
                break

        total_width = left_count + 1 + right_count

        # Check if this is an edge cell (has a non-room neighbor vertically)
        up_in_room = (r - 1, c) in room.cells
        down_in_room = (r + 1, c) in room.cells

        # If it's an edge and the width is less than min_room_width, it's a protrusion
        if (not up_in_room or not down_in_room) and total_width < self.min_room_width:
            return True

        return False

    def _is_horizontal_protrusion_in_room(self, room, r, c):
        """
        Check if this cell is part of a horizontal protrusion smaller than min_room_width within the room.
        """
        # Count continuous room cells up and down
        up_count = 0
        for check_r in range(r - 1, -1, -1):
            if (check_r, c) in room.cells:
                up_count += 1
            else:
                break

        down_count = 0
        for check_r in range(r + 1, self.grid.rows):
            if (check_r, c) in room.cells:
                down_count += 1
            else:
                break

        total_height = up_count + 1 + down_count

        # Check if this is an edge cell (has a non-room neighbor horizontally)
        left_in_room = (r, c - 1) in room.cells
        right_in_room = (r, c + 1) in room.cells

        # If it's an edge and the height is less than min_room_width, it's a protrusion
        if (
            not left_in_room or not right_in_room
        ) and total_height < self.min_room_width:
            return True

        return False

    def _get_largest_connected_component(self, cells):
        """
        Get the largest connected component from a set of cells.
        """
        if not cells:
            return set()

        visited = set()
        components = []

        for cell in cells:
            if cell not in visited:
                # BFS to find this component
                component = set()
                queue = [cell]
                component.add(cell)
                visited.add(cell)

                while queue:
                    r, c = queue.pop(0)

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbor = (r + dr, c + dc)
                        if neighbor in cells and neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)

                components.append(component)

        # Return the largest component
        return max(components, key=len) if components else set()

    def _is_region_connected(self, region_cells):
        """
        Check if a region is connected using BFS.
        """
        if not region_cells:
            return False

        # Start from any cell in the region
        start_cell = next(iter(region_cells))
        visited = set()
        queue = [start_cell]
        visited.add(start_cell)

        while queue:
            r, c = queue.pop(0)

            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)
                if neighbor in region_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(region_cells)

    def _create_l_shapes(self):
        """
        Attempt to merge some adjacent rectangular rooms to create L-shapes.
        """
        merge_attempts = len(self.rooms) // 3  # Try to merge about 1/3 of rooms

        for _ in range(merge_attempts):
            if len(self.rooms) < 2:
                break

            # Pick two random rooms
            room1_idx = random.randint(0, len(self.rooms) - 1)
            room1 = self.rooms[room1_idx]

            # Find adjacent rooms
            adjacent_rooms = self._find_adjacent_rooms(room1)

            if not adjacent_rooms:
                continue

            # Pick a random adjacent room
            room2 = random.choice(adjacent_rooms)
            room2_idx = self.rooms.index(room2)

            # Try to merge them
            merged_cells = room1.cells | room2.cells
            merged_room = Room(merged_cells)

            # Check if merged room is valid (not too large or weird shape)
            if self._is_valid_merged_room(merged_room):
                # Remove old rooms and add merged room
                self.rooms.pop(max(room1_idx, room2_idx))
                self.rooms.pop(min(room1_idx, room2_idx))
                self.rooms.append(merged_room)

    def _find_adjacent_rooms(self, room):
        """
        Find all rooms that are adjacent to the given room.
        """
        adjacent = []

        for other_room in self.rooms:
            if other_room is room:
                continue

            # Check if any cells are adjacent
            for r1, c1 in room.cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (r1 + dr, c1 + dc)
                    if neighbor in other_room.cells:
                        adjacent.append(other_room)
                        break
                if other_room in adjacent:
                    break

        return adjacent

    def _is_valid_merged_room(self, room):
        """
        Check if a merged room has a valid shape (rectangular or L-shaped).
        """
        # Check if it's connected
        if not self._is_region_connected(room.cells):
            return False

        # Check if it's not too large
        if room.get_area() > self.min_room_width * 6:
            return False

        # Check if it's rectangular or L-shaped
        # An L-shape should have area between 50-75% of its bounding box
        area = room.get_area()
        bbox_area = room.get_width() * room.get_height()

        if bbox_area == 0:
            return False

        fill_ratio = area / bbox_area

        # Accept if rectangular (100%) or L-shaped (50-90%)
        return 0.5 <= fill_ratio <= 1.0

    def display_rooms(self):
        """
        Display the rooms with different symbols for each room.
        """
        # Create a grid to display
        display_grid = [
            [" " for _ in range(self.grid.cols)] for _ in range(self.grid.rows)
        ]

        # Assign a number to each room
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        for idx, room in enumerate(self.rooms):
            symbol = symbols[idx % len(symbols)]
            for r, c in room.cells:
                display_grid[r][c] = symbol

        # Print the grid
        for row in display_grid:
            print(" ".join(row))

        # Print room statistics
        print(f"\nTotal rooms: {len(self.rooms)}")
        for idx, room in enumerate(self.rooms):
            shape_type = "Rectangle" if room.is_rectangular() else "L-shape/Other"
            print(
                f"Room {idx}: {room.get_area()} cells, {room.get_width()}x{room.get_height()} bounds, {shape_type}"
            )


class Wall:
    def __init__(self, cells, wall_type, direction):
        """
        cells: list of (r, c) tuples that make up this wall segment
        wall_type: 'internal' or 'external'
        direction: 'horizontal' or 'vertical'
        """
        self.cells = cells
        self.wall_type = wall_type
        self.direction = direction
        self.has_door = False
        self.door_cells = []  # Cells where the door is placed


class Door:
    def __init__(self, cells, room1=None, room2=None, is_entrance=False):
        """
        cells: list of (r, c) tuples where the door is (should be door_size length)
        room1, room2: rooms connected by this door (None if entrance)
        is_entrance: True if this is the main entrance
        """
        self.cells = cells
        self.room1 = room1
        self.room2 = room2
        self.is_entrance = is_entrance


class DoorWindowGenerator:
    def __init__(self, grid, rooms, door_size=2, min_room_width=5):
        self.grid = grid
        self.rooms = rooms
        self.door_size = door_size
        self.min_room_width = min_room_width
        self.doors = []
        self.walls = []

    def generate_doors(self):
        """
        Main method to generate doors.
        """
        # First, identify all walls
        self._identify_walls()

        # Place doors between rooms
        self._place_internal_doors()

        # Ensure all rooms are connected (add more doors if needed)
        self._ensure_connectivity()

        # Place entrance door
        self._place_entrance_door()

        return self.doors

    def _identify_walls(self):
        """
        Identify all wall segments (both internal and external).
        """
        raw_walls = []

        # For each room, find its boundaries
        for room_idx, room in enumerate(self.rooms):
            room_walls = self._get_room_walls(room, room_idx)
            raw_walls.extend(room_walls)

        # Deduplicate overlapping wall segments shared by adjacent rooms
        self.walls = self._consolidate_walls(raw_walls)

    def _consolidate_walls(self, walls):
        """
        Merge duplicate wall segments so shared walls are represented once.
        Keys on direction + ordered cell list, preserving the first wall_type.
        """
        merged = {}
        for wall in walls:
            key = (wall.direction, tuple(wall.cells))
            if key in merged:
                continue
            merged[key] = wall
        return list(merged.values())

    def _get_room_walls(self, room, room_idx):
        """
        Get all wall segments for a room.
        A wall is a continuous segment of cells on the room's edge.
        """
        walls = []
        visited_edges = set()

        # Check each cell in the room
        for r, c in room.cells:
            # Check all 4 directions
            directions = [
                ((-1, 0), "top", "horizontal"),
                ((1, 0), "bottom", "horizontal"),
                ((0, -1), "left", "vertical"),
                ((0, 1), "right", "vertical"),
            ]

            for (dr, dc), edge_name, direction in directions:
                neighbor = (r + dr, c + dc)
                edge_key = (r, c, edge_name)

                if edge_key in visited_edges:
                    continue

                # Check if this is a wall (edge of room)
                is_wall = False
                wall_type = None

                # Check if neighbor is outside the room
                if neighbor not in room.cells:
                    is_wall = True

                    # Determine if it's internal or external
                    if not self.grid.is_valid_coord(neighbor[0], neighbor[1]):
                        wall_type = "external"  # Outside grid bounds
                    elif not self.grid.grid[neighbor[0]][neighbor[1]].is_building_cell:
                        wall_type = "external"  # Outside building
                    else:
                        # Check if it's adjacent to another room
                        for other_room in self.rooms:
                            if other_room is room and neighbor in other_room.cells:
                                wall_type = "internal"
                                break
                        if wall_type is None:
                            wall_type = "internal"

                if is_wall:
                    # Find the continuous wall segment
                    wall_segment = self._trace_wall_segment(
                        room, r, c, edge_name, direction, visited_edges
                    )

                    if wall_segment:
                        wall = Wall(wall_segment, wall_type, direction)
                        walls.append(wall)

        return walls

    def _trace_wall_segment(
        self, room, start_r, start_c, edge_name, direction, visited_edges
    ):
        """
        Trace a continuous wall segment starting from a given cell.
        """
        segment = []

        if direction == "horizontal":
            # Trace left and right
            current_c = start_c

            # Trace right
            while True:
                edge_key = (start_r, current_c, edge_name)
                if edge_key in visited_edges:
                    break
                if (start_r, current_c) not in room.cells:
                    break

                # Check if this cell has the same edge type
                if edge_name == "top":
                    neighbor = (start_r - 1, current_c)
                else:  # bottom
                    neighbor = (start_r + 1, current_c)

                if neighbor in room.cells:
                    break

                segment.append((start_r, current_c))
                visited_edges.add(edge_key)
                current_c += 1

            # Trace left
            current_c = start_c - 1
            while True:
                edge_key = (start_r, current_c, edge_name)
                if edge_key in visited_edges:
                    break
                if (start_r, current_c) not in room.cells:
                    break

                # Check if this cell has the same edge type
                if edge_name == "top":
                    neighbor = (start_r - 1, current_c)
                else:  # bottom
                    neighbor = (start_r + 1, current_c)

                if neighbor in room.cells:
                    break

                segment.append((start_r, current_c))
                visited_edges.add(edge_key)
                current_c -= 1

            # Sort by column
            segment.sort(key=lambda x: x[1])

        else:  # vertical
            # Trace up and down
            current_r = start_r

            # Trace down
            while True:
                edge_key = (current_r, start_c, edge_name)
                if edge_key in visited_edges:
                    break
                if (current_r, start_c) not in room.cells:
                    break

                # Check if this cell has the same edge type
                if edge_name == "left":
                    neighbor = (current_r, start_c - 1)
                else:  # right
                    neighbor = (current_r, start_c + 1)

                if neighbor in room.cells:
                    break

                segment.append((current_r, start_c))
                visited_edges.add(edge_key)
                current_r += 1

            # Trace up
            current_r = start_r - 1
            while True:
                edge_key = (current_r, start_c, edge_name)
                if edge_key in visited_edges:
                    break
                if (current_r, start_c) not in room.cells:
                    break

                # Check if this cell has the same edge type
                if edge_name == "left":
                    neighbor = (current_r, start_c - 1)
                else:  # right
                    neighbor = (current_r, start_c + 1)

                if neighbor in room.cells:
                    break

                segment.append((current_r, start_c))
                visited_edges.add(edge_key)
                current_r -= 1

            # Sort by row
            segment.sort(key=lambda x: x[0])

        return segment

    def _place_internal_doors(self):
        """
        Place doors between adjacent rooms on internal walls.
        """
        # Find all internal walls that are long enough for a door
        internal_walls = [
            w
            for w in self.walls
            if w.wall_type == "internal" and len(w.cells) >= self.door_size
        ]

        # For each pair of adjacent rooms, try to place a door
        placed_doors = set()

        for wall in internal_walls:
            if wall.has_door:
                continue

            # Find which rooms this wall separates
            room_pair = self._find_rooms_separated_by_wall(wall)

            if room_pair and room_pair not in placed_doors:
                # Try to place a door on this wall
                door_cells = self._find_door_position(wall)

                if door_cells:
                    door = Door(door_cells, room1=room_pair[0], room2=room_pair[1])
                    self.doors.append(door)
                    wall.has_door = True
                    wall.door_cells = door_cells
                    placed_doors.add(room_pair)
                    placed_doors.add((room_pair[1], room_pair[0]))  # Add reverse pair

    def _find_rooms_separated_by_wall(self, wall):
        """
        Find which two rooms are separated by this wall.
        """
        if not wall.cells:
            return None

        # Get a cell from the wall
        sample_cell = wall.cells[0]
        r, c = sample_cell

        # Find the room this cell belongs to
        room1 = None
        for room in self.rooms:
            if sample_cell in room.cells:
                room1 = room
                break

        if not room1:
            return None

        # Check neighbors to find the adjacent room
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)

            if neighbor not in room1.cells:
                # Check if this neighbor belongs to another room
                for room2 in self.rooms:
                    if room2 is not room1 and neighbor in room2.cells:
                        return (room1, room2)

        return None

    def _find_door_position(self, wall):
        """
        Find a good position for a door on a wall.
        Returns a list of door_size cells where the door should be placed.
        """
        if len(wall.cells) < self.door_size:
            return None

        # Try to place door in the middle of the wall
        wall_length = len(wall.cells)

        # Calculate middle position
        start_idx = (wall_length - self.door_size) // 2

        # Get door cells
        door_cells = wall.cells[start_idx : start_idx + self.door_size]

        return door_cells

    def _ensure_connectivity(self):
        """
        Ensure all rooms are connected via doors.
        Uses Union-Find to check connectivity and adds doors where needed.
        """
        # Build a graph of room connections
        room_connections = {i: set() for i in range(len(self.rooms))}

        for door in self.doors:
            if door.room1 and door.room2:
                idx1 = self.rooms.index(door.room1)
                idx2 = self.rooms.index(door.room2)
                room_connections[idx1].add(idx2)
                room_connections[idx2].add(idx1)

        # Find connected components using BFS
        visited = set()
        components = []

        for i in range(len(self.rooms)):
            if i not in visited:
                component = set()
                queue = [i]
                component.add(i)
                visited.add(i)

                while queue:
                    current = queue.pop(0)
                    for neighbor in room_connections[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)

                components.append(component)

        # If there's more than one component, connect them
        while len(components) > 1:
            # Connect the first two components
            comp1 = components[0]
            comp2 = components[1]

            # Find the closest rooms between these components
            best_wall = None
            best_distance = float("inf")

            for wall in self.walls:
                if wall.wall_type != "internal" or wall.has_door:
                    continue
                if len(wall.cells) < self.door_size:
                    continue

                room_pair = self._find_rooms_separated_by_wall(wall)
                if not room_pair:
                    continue

                idx1 = self.rooms.index(room_pair[0])
                idx2 = self.rooms.index(room_pair[1])

                # Check if this wall connects the two components
                if (idx1 in comp1 and idx2 in comp2) or (
                    idx1 in comp2 and idx2 in comp1
                ):
                    # Calculate "distance" (for now, just use wall length)
                    distance = len(wall.cells)
                    if distance < best_distance:
                        best_distance = distance
                        best_wall = wall

            if best_wall:
                # Place a door on this wall
                door_cells = self._find_door_position(best_wall)
                room_pair = self._find_rooms_separated_by_wall(best_wall)

                if door_cells and room_pair:
                    door = Door(door_cells, room1=room_pair[0], room2=room_pair[1])
                    self.doors.append(door)
                    best_wall.has_door = True
                    best_wall.door_cells = door_cells

                    # Merge the components
                    merged = comp1 | comp2
                    components = [merged] + components[2:]
            else:
                # Can't connect, break to avoid infinite loop
                break

    def _place_entrance_door(self):
        """
        Place an entrance door on an external wall.
        """
        # Find all external walls that are long enough
        external_walls = [
            w
            for w in self.walls
            if w.wall_type == "external" and len(w.cells) >= self.door_size
        ]

        if not external_walls:
            return

        # Prefer walls on the bottom or front of the building
        # For now, just pick a random external wall
        entrance_wall = random.choice(external_walls)

        door_cells = self._find_door_position(entrance_wall)

        if door_cells:
            # Find which room this entrance connects to
            sample_cell = door_cells[0]
            entrance_room = None
            for room in self.rooms:
                if sample_cell in room.cells:
                    entrance_room = room
                    break

            door = Door(door_cells, room1=entrance_room, is_entrance=True)
            self.doors.append(door)
            entrance_wall.has_door = True
            entrance_wall.door_cells = door_cells

    def display_with_doors(self):
        """
        Display the floor plan with rooms and doors.
        """
        # Create a grid to display
        display_grid = [
            [" " for _ in range(self.grid.cols)] for _ in range(self.grid.rows)
        ]

        # Assign symbols to rooms
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        for idx, room in enumerate(self.rooms):
            symbol = symbols[idx % len(symbols)]
            for r, c in room.cells:
                display_grid[r][c] = symbol

        # Mark doors
        for door in self.doors:
            for r, c in door.cells:
                if door.is_entrance:
                    display_grid[r][c] = "E"  # Entrance
                else:
                    display_grid[r][c] = "D"  # Door

        # Print the grid
        for row in display_grid:
            print(" ".join(row))

        # Print statistics
        print(f"\nTotal rooms: {len(self.rooms)}")
        print(
            f"Total doors: {len(self.doors)} ({sum(1 for d in self.doors if d.is_entrance)} entrance)"
        )


# After creating rooms
room_divider = RoomDivider(base_grid, min_room_width=min_room_width)
rooms = room_divider.divide_into_rooms()

# Generate doors
door_window_gen = DoorWindowGenerator(
    base_grid, rooms, door_size=door_size, min_room_width=min_room_width
)
doors = door_window_gen.generate_doors()

# Display the result
door_window_gen.display_with_doors()


import cv2
import numpy as np


class FloorPlanVisualizer:
    def __init__(self, grid, rooms, doors, cell_size=20):
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
            (230, 190, 255),  # Lavender
            (255, 220, 190),  # Peach
            (190, 255, 230),  # Mint
            (255, 240, 190),  # Light yellow
            (255, 190, 220),  # Pink
            (190, 230, 255),  # Light blue
            (220, 255, 190),  # Light green
            (240, 190, 255),  # Light purple
            (190, 255, 255),  # Cyan
            (255, 210, 190),  # Coral
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


import matplotlib.pyplot as plt

# After generating doors
visualizer = FloorPlanVisualizer(base_grid, rooms, doors, cell_size=30)

# Create the floor plan
floor_plan_img = visualizer.create_floor_plan()

# Optionally add room labels
floor_plan_img = visualizer.add_room_labels(floor_plan_img)

# Visualize in Google Colab
# Method 1: Using cv2_imshow (simpler)
print("Floor Plan Visualization:")
cv2.imshow("Floor Plan", floor_plan_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Method 2: Using matplotlib (more control over size)
# plt.figure(figsize=(15, 12))
# # Convert BGR to RGB for matplotlib
# floor_plan_rgb = cv2.cvtColor(floor_plan_img, cv2.COLOR_BGR2RGB)
# plt.imshow(floor_plan_rgb)
# plt.axis('off')
# plt.title('Floor Plan', fontsize=16)
# plt.tight_layout()
# plt.show()

# Save the image (optional)
visualizer.save_image(floor_plan_img, "my_floor_plan.png")
print("Floor plan saved to 'my_floor_plan.png'")
