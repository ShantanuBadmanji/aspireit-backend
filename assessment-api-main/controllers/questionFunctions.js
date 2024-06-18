// import asyncHandler from "express-async-handler";
import Question from "../models/questionModel.js";


// const getQuestions = asyncHandler(async (req, res) => {
//   const questions = await Question.find({});
//   if (questions) {
//     res.status(200).json(questions);
//   } else {
//     res.status(400);
//     throw new Error("No questions found");
//   }
// });

// const createQuestion = asyncHandler(async (req, res) => {
//   const { assessmentId } = req.params;
//   const { question, accepted_keywords, difficulty, topic } = req.body;

//   if (!question) {
//     res.status(400);
//     throw new Error("Insufficient details provided");
//   }

//   const newQuestion = new Question({
//     question: question.toLowerCase(),
//     // accepted_keywords,
//     difficulty,
//     topic,
//   });

//   const savedQuestion = await newQuestion.save();
//   // const assessment = await assessmentModel.findById(assessmentId);
//   //   if (assessment) {
//   //       assessment.questions.push(savedQuestion._id);
//   //       await assessment.save();
//   //   } else {
//   //       res.status(400);
//   //       throw new Error("Assessment not found");
//   //   }

//   if (savedQuestion) {
//     res.status(200).json(savedQuestion);
//   } else {
//     res.status(400);
//     throw new Error("Unable to create new question");
//   }
// });

// const deleteQuestion = asyncHandler(async (req, res) => {
//   const questionId = req.params.id;
//   const deletedQuestion = await Question.findByIdAndDelete(questionId);
//   if (deletedQuestion) {
//     res.status(200).json({ message: "Question deleted successfully" });
//   } else {
//     res.status(400);
//     throw new Error("Unable to delete question");
//   }
// });

// export { getQuestions, createQuestion, deleteQuestion };
// controllers/questionController.js

const createQuestions = async (req, res) => {
  try {
      const { questions } = req.body;

      // Validate incoming data
      if (!Array.isArray(questions)) {
          return res.status(400).json({ error: 'Questions should be an array' });
      }

      // Validate each question
      questions.forEach(question => {
          if (typeof question.question !== 'string' || question.question.trim() === '') {
              throw new Error('Question must be a non-empty string');
          }
          if (!question.topic || !['dsa', 'oops', 'ai', 'nlp', 'ml', 'opencv', 'dbms', 'os', 'cloud','java'].includes(question.topic)) {
              throw new Error('Invalid topic');
          }
      });

      // Save questions to MongoDB
      const savedQuestions = await Question.insertMany(questions);

      res.status(201).json(savedQuestions);
  } catch (error) {
      console.error('Error saving questions:', error.message);
      res.status(500).json({ error: 'Failed to save questions' });
  }
};

export { createQuestions };
