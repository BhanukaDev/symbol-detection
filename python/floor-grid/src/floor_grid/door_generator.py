import random
from typing import List, Optional, Set
from .models import Grid, Room, Wall, Door


class DoorWindowGenerator:
    def __init__(self, grid: Grid, rooms: List[Room], door_size=2, min_room_width=5):
        self.grid = grid
        self.rooms = rooms
        self.door_size = door_size
        self.min_room_width = min_room_width
        self.doors = []
        self.walls = []
        self.placed_door_locations = set()  # Track all placed door cell locations

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
        Only places doors on walls with sufficient length.
        Prevents duplicate doors at the same location.
        """
        # Find all internal walls that are long enough for a door
        internal_walls = [
            w
            for w in self.walls
            if w.wall_type == "internal" and len(w.cells) >= self.door_size
        ]

        # For each pair of adjacent rooms, try to place a door
        placed_room_pairs = set()

        for wall in internal_walls:
            if wall.has_door:
                continue

            # Find which rooms this wall separates
            room_pair = self._find_rooms_separated_by_wall(wall)

            if room_pair:
                # Create a canonical room pair representation
                room1_idx = self.rooms.index(room_pair[0])
                room2_idx = self.rooms.index(room_pair[1])
                pair_key = tuple(sorted([room1_idx, room2_idx]))

                # Skip if we've already placed a door for this room pair
                if pair_key in placed_room_pairs:
                    continue

                # Try to place a door on this wall
                door_cells = self._find_door_position(wall)

                if door_cells:
                    # Check if door location is not already occupied
                    door_location = tuple(sorted(door_cells))
                    if door_location not in self.placed_door_locations:
                        door = Door(door_cells, room1=room_pair[0], room2=room_pair[1])
                        self.doors.append(door)
                        wall.has_door = True
                        wall.door_cells = door_cells
                        placed_room_pairs.add(pair_key)
                        # Record this door location
                        for cell in door_cells:
                            self.placed_door_locations.add(cell)

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
        Uses BFS to check connectivity and adds doors where needed.
        Only places doors on walls with sufficient size.
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

            # Find the best wall between these components
            best_wall = None
            best_distance = float("inf")

            for wall in self.walls:
                # Only consider walls with sufficient size and no existing door
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
                    # Prefer longer walls (better for door placement)
                    distance = len(wall.cells)
                    if distance > best_distance:
                        best_distance = distance
                        best_wall = wall

            if best_wall:
                # Place a door on this wall
                door_cells = self._find_door_position(best_wall)
                room_pair = self._find_rooms_separated_by_wall(best_wall)

                if door_cells and room_pair:
                    # Check if door location is not already occupied
                    door_location = tuple(sorted(door_cells))
                    if door_location not in self.placed_door_locations:
                        door = Door(door_cells, room1=room_pair[0], room2=room_pair[1])
                        self.doors.append(door)
                        best_wall.has_door = True
                        best_wall.door_cells = door_cells
                        # Record this door location
                        for cell in door_cells:
                            self.placed_door_locations.add(cell)

                        # Merge the components
                        merged = comp1 | comp2
                        components = [merged] + components[2:]
                    else:
                        # Door location conflicts, can't connect
                        break
                else:
                    # Can't place door, break to avoid infinite loop
                    break
            else:
                # Can't find suitable wall, break to avoid infinite loop
                break

    def _place_entrance_door(self):
        """
        Place an entrance door on an external wall.
        Only places doors on walls with sufficient length.
        Avoids placing doors at locations already occupied.
        """
        # Find all external walls that are long enough
        external_walls = [
            w
            for w in self.walls
            if w.wall_type == "external"
            and len(w.cells) >= self.door_size
            and not w.has_door
        ]

        if not external_walls:
            return

        # Try each external wall until we find a valid placement
        random.shuffle(external_walls)
        for entrance_wall in external_walls:
            door_cells = self._find_door_position(entrance_wall)

            if door_cells:
                # Check if door location is not already occupied
                door_location = tuple(sorted(door_cells))
                if door_location not in self.placed_door_locations:
                    # Find which room this entrance connects to
                    sample_cell = door_cells[0]
                    entrance_room = None
                    for room in self.rooms:
                        if sample_cell in room.cells:
                            entrance_room = room
                            break

                    if entrance_room:
                        door = Door(door_cells, room1=entrance_room, is_entrance=True)
                        self.doors.append(door)
                        entrance_wall.has_door = True
                        entrance_wall.door_cells = door_cells
                        # Record this door location
                        for cell in door_cells:
                            self.placed_door_locations.add(cell)
                        return  # Successfully placed entrance door

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
                    display_grid[r][c] = "D"  # Interna door

        # Print the grid
        for row in display_grid:
            print(" ".join(row))

        # Print door statistics
        print(f"\nTotal doors: {len(self.doors)}")
        entrance_count = sum(1 for d in self.doors if d.is_entrance)
        print(f"Entrance doors: {entrance_count}")
        print(f"Internal doors: {len(self.doors) - entrance_count}")
