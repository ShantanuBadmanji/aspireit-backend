import mongoose from "mongoose";

const responseSchema = new mongoose.Schema(
  {
    candidate_id: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Candidate",
      required: true,
    },
    assessment_id: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Assessment",
      required: true,
    },
    responses: {
      // array of audios
      type: [{ question: String, recording: Buffer }],
      default: [],
    },
  },
  { timestamps: true }
);

const responseModel = mongoose.model("Response", responseSchema);

export default responseModel;
