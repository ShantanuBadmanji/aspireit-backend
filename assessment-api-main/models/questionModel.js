import mongoose from "mongoose";

const questionSchema = new mongoose.Schema({
    question: {
        type: String,
        required: true,
    },
    topic: {
        type: String,
        enum: ['dsa', 'oops', 'frontend', 'backend', 'ml', 'networks', 'dbms', 'os', 'cloud','java'],
        default: 'dsa',
    },
    // difficulty: {
    //     type: String,
    //     enum: ['easy', 'medium', 'hard'],
    //     default: 'medium',
  
}, { timestamps: true });


const Question = mongoose.model('Question', questionSchema);

export default Question;