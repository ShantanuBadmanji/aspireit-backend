import asyncHandler from "express-async-handler";
import Question from "../models/questionModel.js";
import assessmentModel from "../models/assessmentModel.js";

const getQuestions = asyncHandler(async (req, res) => {
  const questions = await Question.find({});
  if (questions) {
    res.status(200).json(questions);
  } else {
    res.status(400);
    throw new Error("No questions found");
  }
});

const createQuestion = asyncHandler(async (req, res) => {
  const { assessmentId } = req.params;
  const { question, accepted_keywords, difficulty, topic } = req.body;

  if (!question) {
    res.status(400);
    throw new Error("Insufficient details provided");
  }

  const newQuestion = new Question({
    question: question.toLowerCase(),
    // accepted_keywords,
    difficulty,
    topic,
  });

  const savedQuestion = await newQuestion.save();
  // const assessment = await assessmentModel.findById(assessmentId);
  //   if (assessment) {
  //       assessment.questions.push(savedQuestion._id);
  //       await assessment.save();
  //   } else {
  //       res.status(400);
  //       throw new Error("Assessment not found");
  //   }

  if (savedQuestion) {
    res.status(200).json(savedQuestion);
  } else {
    res.status(400);
    throw new Error("Unable to create new question");
  }
});

const deleteQuestion = asyncHandler(async (req, res) => {
  const questionId = req.params.id;
  const deletedQuestion = await Question.findByIdAndDelete(questionId);
  if (deletedQuestion) {
    res.status(200).json({ message: "Question deleted successfully" });
  } else {
    res.status(400);
    throw new Error("Unable to delete question");
  }
});

export { getQuestions, createQuestion, deleteQuestion };
