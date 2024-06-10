import express from "express";
import {
  getResponses,
  createResponse,
  deleteResponse,
  updateResponse,
} from "../controllers/responseFunctions.js";
import { upload } from "../lib/multer.js";

const router = express.Router();

router.route("/").get(getResponses).post(createResponse);
router
  .route("/:responseId")
  .delete(deleteResponse)
  .patch(upload.single("audio_file"), updateResponse);

export default router;
