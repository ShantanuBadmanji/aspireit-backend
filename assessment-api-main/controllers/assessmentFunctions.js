import asyncHandler from "express-async-handler";
import Assessment from "../models/assessmentModel.js";
import mongoose from "mongoose";

const createAssessment = asyncHandler(async (req, res) => {
  const { questions, role, duration } = req.body;
  const assessment = new Assessment({ questions, role, duration });
  const createdAssessment = await assessment.save();
  if (createAssessment) {
    res.status(200).json(createdAssessment);
  } else {
    res.status(400);
    throw new Error({ message: "Assessment creation failed" });
  }
});

const getAssessments = asyncHandler(async (req, res) => {
  const { assessmentId } = req.query;
  if (assessmentId && !mongoose.Types.ObjectId.isValid(assessmentId)) {
    res.status(400);
    throw new Error({ message: "Invalid assessment ID" });
  }
  const queryFilter = assessmentId ? { _id: assessmentId } : {};
  const assessments = await Assessment.find(queryFilter);
  if (assessments) {
    console.log(assessments);
    res.status(200).json(assessments);
  } else {
    res.status(400);
    throw new Error({ message: "No assessments found" });
  }
});

const updateAssessment = asyncHandler(async (req, res) => {
  const {assessmentId} = req.params;
  console.log({assessmentId});
  if (!mongoose.Types.ObjectId.isValid(assessmentId)) {
    res.status(400);
    throw new Error("Invalid assessment ID");
  }

  const { questions } = req.body;
  console.log({questions});
  // let updatedAssessment = await Assessment.findById(assessmentId);
  // updatedAssessment.questions = questions;
  // updatedAssessment = await updatedAssessment.save();
  const updatedAssessment = await Assessment.findByIdAndUpdate(
    assessmentId,
    { questions },
    { new: true, runValidators: true }
  );
  if (updatedAssessment) {
    res.status(200).json(updatedAssessment);
  } else {
    res.status(400);
    throw new Error("Assessment update failed");
  }
});

const deleteAssessment = asyncHandler(async (req, res) => {
  const {assessmentId} = req.params;
  if (!mongoose.Types.ObjectId.isValid(assessmentId)) {
    res.status(400);
    throw new Error("Invalid assessment ID");
  }
  const deletedAssessment = await Assessment.findByIdAndDelete(assessmentId);
  if (deletedAssessment) {
    res.status(200).json({ message: "Assessment deleted successfully" });
  } else {
    res.status(400);
    throw new Error({ message: "Assessment deletion failed" });
  }
});

export { createAssessment, getAssessments, updateAssessment, deleteAssessment };
