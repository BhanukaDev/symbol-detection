import { AppBar, Toolbar, Typography, Button, Box, Container } from "@mui/material";
import { Link, useLocation } from "react-router-dom";
import BoltIcon from "@mui/icons-material/Bolt";

export default function Navbar() {
  const location = useLocation();

  const navItems = [
    { label: "Detector", path: "/" },
    { label: "About", path: "/about" },
  ];

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        bgcolor: "rgba(255,255,255,0.85)",
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid",
        borderColor: "divider",
      }}
    >
      <Container maxWidth="lg">
        <Toolbar disableGutters sx={{ gap: 1 }}>
          <BoltIcon sx={{ color: "secondary.main", fontSize: 32 }} />
          <Typography
            variant="h6"
            component={Link}
            to="/"
            sx={{
              color: "primary.main",
              textDecoration: "none",
              fontWeight: 800,
              mr: 4,
            }}
          >
            SymbolAI
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          {navItems.map((item) => (
            <Button
              key={item.path}
              component={Link}
              to={item.path}
              sx={{
                color: location.pathname === item.path ? "primary.main" : "text.secondary",
                fontWeight: location.pathname === item.path ? 700 : 500,
                borderBottom: location.pathname === item.path ? "2px solid" : "2px solid transparent",
                borderColor: location.pathname === item.path ? "secondary.main" : "transparent",
                borderRadius: 0,
                px: 2,
                "&:hover": { bgcolor: "transparent", color: "primary.main" },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Toolbar>
      </Container>
    </AppBar>
  );
}
