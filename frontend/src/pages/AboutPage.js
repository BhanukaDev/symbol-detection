import {
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  Chip,
  Divider,
  Stack,
} from "@mui/material";
import ArchitectureIcon from "@mui/icons-material/Architecture";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import DatasetIcon from "@mui/icons-material/Dataset";
import SpeedIcon from "@mui/icons-material/Speed";
import CropIcon from "@mui/icons-material/Crop";
import BoltIcon from "@mui/icons-material/Bolt";

const SYMBOL_CLASSES = [
  { name: "Light", color: "#FF6B6B" },
  { name: "Duplex Receptacle", color: "#4ECDC4" },
  { name: "Single-pole, one-way switch", color: "#45B7D1" },
  { name: "Two-pole, one-way switch", color: "#96CEB4" },
  { name: "Three-pole, one-way switch", color: "#FFEAA7" },
  { name: "Two-way switch", color: "#DDA0DD" },
  { name: "Junction Box", color: "#FF8C42" },
];

function InfoCard({ icon, title, children }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        height: "100%",
        border: "1px solid",
        borderColor: "divider",
        transition: "box-shadow 0.2s, transform 0.2s",
        "&:hover": {
          boxShadow: "0 8px 32px rgba(0,0,0,0.08)",
          transform: "translateY(-2px)",
        },
      }}
    >
      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2 }}>
        <Box
          sx={{
            p: 1,
            borderRadius: 2,
            bgcolor: "primary.main",
            color: "white",
            display: "flex",
          }}
        >
          {icon}
        </Box>
        <Typography variant="h6">{title}</Typography>
      </Stack>
      {children}
    </Paper>
  );
}

export default function AboutPage() {
  return (
    <Box sx={{ py: 6, minHeight: "calc(100vh - 64px)" }}>
      <Container maxWidth="lg">
        {/* Hero */}
        <Box sx={{ textAlign: "center", mb: 8 }}>
          <Typography
            variant="h3"
            gutterBottom
            sx={{
              background: "linear-gradient(135deg, #1a237e 0%, #00bfa5 100%)",
              backgroundClip: "text",
              WebkitBackgroundClip: "text",
              color: "transparent",
            }}
          >
            How It Works
          </Typography>
          <Typography
            variant="h6"
            color="text.secondary"
            sx={{ maxWidth: 700, mx: "auto" }}
          >
            An AI-powered system that detects electrical symbols in architectural
            floor plans using deep learning object detection.
          </Typography>
        </Box>

        {/* Architecture Cards */}
        <Grid container spacing={3} sx={{ mb: 6 }}>
          <Grid size={{ xs: 12, md: 6 }}>
            <InfoCard
              icon={<ArchitectureIcon />}
              title="Model Architecture"
            >
              <Typography variant="body1" color="text.secondary" gutterBottom>
                We use <strong>Faster R-CNN</strong> with a{" "}
                <strong>ResNet-50 + FPN</strong> (Feature Pyramid Network)
                backbone. This is a two-stage object detector:
              </Typography>
              <Stack spacing={1} sx={{ mt: 2 }}>
                <Box>
                  <Typography variant="subtitle2">
                    1. Region Proposal Network (RPN)
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Scans the image at multiple scales and proposes candidate
                    regions that might contain symbols.
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="subtitle2">
                    2. Detection Head
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Classifies each proposal into one of 7 symbol classes and
                    refines the bounding box coordinates.
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="subtitle2">
                    Feature Pyramid Network (FPN)
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Multi-scale feature maps allow the model to detect both small
                    symbols (switches) and larger ones (junction boxes) effectively.
                  </Typography>
                </Box>
              </Stack>
            </InfoCard>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <InfoCard icon={<ModelTrainingIcon />} title="Training">
              <Typography variant="body1" color="text.secondary" gutterBottom>
                The model is trained on synthetically generated floor plan
                images with pixel-perfect annotations using <strong>CIoU
                Loss</strong> (Complete IoU), which improves bounding box
                regression by considering overlap area, center distance, and
                aspect ratio consistency.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
                Training is performed on an <strong>NVIDIA A100 GPU</strong> with
                a cosine-annealing learning rate schedule and SGD optimizer for
                stable convergence.
              </Typography>
            </InfoCard>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <InfoCard icon={<DatasetIcon />} title="Synthetic Dataset">
              <Typography variant="body1" color="text.secondary" gutterBottom>
                Electrical symbol datasets for object detection are extremely
                scarce. We built a custom synthetic data pipeline that generates
                realistic training images with perfect ground-truth annotations:
              </Typography>
              <Stack spacing={1} sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Grid-based floor plans</strong> — rooms with walls,
                  doors, and realistic proportions are generated procedurally.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>Symbol placement</strong> — electrical symbols are
                  placed at realistic locations within rooms (on walls for
                  switches, ceiling for lights).
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>Augmentations</strong> — discrete rotations (0°, 90°,
                  180°, 270°), scale variation, and distractor objects
                  (furniture) to improve robustness.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>Image effects</strong> — noise, blur, and contrast
                  adjustments simulate real scanned documents.
                </Typography>
              </Stack>
            </InfoCard>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <InfoCard icon={<CropIcon />} title="SAHI Tiling">
              <Typography variant="body1" color="text.secondary" gutterBottom>
                Real floor plans are often very large (3000×4000+ pixels). Small
                symbols would be missed at low resolution.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
                We use <strong>SAHI</strong> (Slicing Aided Hyper Inference) to
                automatically:
              </Typography>
              <Stack spacing={1} sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  <strong>1.</strong> Slice the image into overlapping 512×512
                  tiles
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>2.</strong> Run detection on each tile independently
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>3.</strong> Merge results using NMS (Non-Maximum
                  Suppression) to remove duplicate detections at tile boundaries
                </Typography>
              </Stack>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mt: 2 }}
              >
                This is triggered automatically for images larger than 1024px on
                any side.
              </Typography>
            </InfoCard>
          </Grid>
        </Grid>

        {/* Detectable Symbols */}
        <Paper
          elevation={0}
          sx={{ p: 4, border: "1px solid", borderColor: "divider", mb: 6 }}
        >
          <Stack
            direction="row"
            spacing={1.5}
            alignItems="center"
            sx={{ mb: 3 }}
          >
            <BoltIcon sx={{ color: "secondary.main", fontSize: 28 }} />
            <Typography variant="h5">Detectable Symbols</Typography>
          </Stack>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            The model recognizes 7 classes of common electrical symbols found in
            architectural drawings:
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {SYMBOL_CLASSES.map((cls) => (
              <Chip
                key={cls.name}
                label={cls.name}
                sx={{
                  bgcolor: cls.color,
                  color: "#000",
                  fontWeight: 600,
                  fontSize: "0.9rem",
                  py: 2.5,
                  px: 1,
                }}
              />
            ))}
          </Stack>
        </Paper>

        {/* Pipeline */}
        <Paper
          elevation={0}
          sx={{ p: 4, border: "1px solid", borderColor: "divider", mb: 6 }}
        >
          <Stack
            direction="row"
            spacing={1.5}
            alignItems="center"
            sx={{ mb: 3 }}
          >
            <SpeedIcon sx={{ color: "secondary.main", fontSize: 28 }} />
            <Typography variant="h5">Inference Pipeline</Typography>
          </Stack>
          <Stack
            direction={{ xs: "column", md: "row" }}
            divider={
              <Divider
                orientation="vertical"
                flexItem
                sx={{ display: { xs: "none", md: "block" } }}
              />
            }
            spacing={3}
          >
            {[
              {
                step: "1",
                title: "Upload",
                desc: "Image sent to the API hosted on Hugging Face Spaces",
              },
              {
                step: "2",
                title: "Preprocess",
                desc: "Resize to 512px, letterbox padding, normalize to ImageNet stats",
              },
              {
                step: "3",
                title: "Detect",
                desc: "Faster R-CNN runs inference (with SAHI tiling for large images)",
              },
              {
                step: "4",
                title: "Postprocess",
                desc: "NMS filtering, confidence thresholding (≥50%), coordinate mapping",
              },
              {
                step: "5",
                title: "BOQ",
                desc: "Group detections by class, count quantities, generate bill of quantities",
              },
            ].map((item) => (
              <Box key={item.step} sx={{ flex: 1, textAlign: "center" }}>
                <Box
                  sx={{
                    width: 36,
                    height: 36,
                    borderRadius: "50%",
                    bgcolor: "secondary.main",
                    color: "white",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 700,
                    mx: "auto",
                    mb: 1,
                  }}
                >
                  {item.step}
                </Box>
                <Typography variant="subtitle1" fontWeight={600}>
                  {item.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {item.desc}
                </Typography>
              </Box>
            ))}
          </Stack>
        </Paper>

        {/* Footer Note */}
        <Box sx={{ textAlign: "center", py: 4 }}>
          <Typography variant="body2" color="text.disabled">
            Built with PyTorch, FastAPI, React, and Hugging Face Spaces
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}
