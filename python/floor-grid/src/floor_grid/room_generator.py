import random
from typing import Set, Tuple
from .models import Grid, Room


class RoomDivider:
    def __init__(self, grid: Grid, min_room_width=5):
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

    def _is_region_connected(self, region_cells: Set[Tuple[int, int]]):
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
