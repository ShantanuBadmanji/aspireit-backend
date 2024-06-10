import asyncHandler from "express-async-handler";
import responseModel from "../models/responseModel.js";
import fs from "fs";
import mongoose from "mongoose";

const createResponse = asyncHandler(async (req, res) => {
  const { candidate_id, assessment_id } = req.body;
  const response = new responseModel({
    candidate_id,
    assessment_id,
  });
  const createdResponse = await response.save();
  if (createdResponse) {
    res.status(201).json(createdResponse);
  } else {
    res.status(400);
    throw new Error({ message: "Response creation failed" });
  }
});

const getResponses = asyncHandler(async (req, res) => {
  const { responseId } = req.query;
  const queryFilter = { _id: responseId };
  const responses = await responseModel.find(queryFilter);
  if (responses) {
    res.status(200).json(responses);
  } else {
    res.status(400);
    throw new Error({ message: "No responses found" });
  }
});

const updateResponse = asyncHandler(async (req, res) => {
  const { responseId } = req.params;
  const audioFile = req.file;
  const { question } = req.body;
  console.log({ responseId, audioFile, question });

  if (!mongoose.Types.ObjectId.isValid(responseId)) {
    res.status(400);
    throw new Error("Invalid response id");
  }
  if (!audioFile) {
    res.status(400);
    throw new Error("No response file provided");
  }
  if (!audioFile || !question) {
    res.status(400);
    throw new Error("Insufficient details provided");
  }

  try {
    let buffer;
    try {
      buffer = fs.readFileSync(audioFile.path);
    } catch (error) {
      throw new Error(`Error reading file: ${error.message}`);
    }

    const updatedResponse = await responseModel.findByIdAndUpdate(
      responseId,
      {
        $push: {
          responses: {
            question: question,
            recording: buffer,
          },
        },
      },
      { new: true, runValidators: true }
    );
    // let updatedResponse = await responseModel.findById(responseId);
    // updatedResponse.responses.push({question: question, recording: buffer});
    // updatedResponse = await updatedResponse.save();
    console.log(updatedResponse);
    if (!updatedResponse) {
      res.status(400);
      throw new Error("Response update failed");
    }

    res.status(200).json(updatedResponse);
  } catch (error) {
    console.log(error);
    throw new Error(error.message ?? "Error updating response");
  } finally {
    fs.unlinkSync(audioFile.path);
  }
});

const deleteResponse = asyncHandler(async (req, res) => {
  const { responseId } = req.params;
  const deletedResponse = await responseModel.findByIdAndDelete(responseId);
  if (deletedResponse) {
    res.status(200).json({ message: "Response deleted successfully" });
  } else {
    res.status(400);
    throw new Error("Unable to delete response");
  }
});

export { createResponse, getResponses, updateResponse, deleteResponse };
