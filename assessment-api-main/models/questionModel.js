import mongoose from "mongoose";

const questionSchema = new mongoose.Schema({
    topic: { type: String, required: true },
    content: { type: String, required: true }
}, { timestamps: true });


const Question = mongoose.model('Question', questionSchema);

export default Question;