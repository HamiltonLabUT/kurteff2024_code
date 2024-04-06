//
//  SelectOptionsViewController.swift
//  CV
//
//  Created by Liberty Hamilton on 3/15/18.
//  Copyright © 2018 Liberty Hamilton. All rights reserved.
//

import UIKit

class SelectOptionsViewController: UIViewController, UIPickerViewDelegate, UIPickerViewDataSource {
    
    // The menu of stimulus sets to choose from
    @IBOutlet var stimulusSet: UIPickerView!
    
    var stimulusSetList: [String] = [String]()
    var fileList = [String]() // This will be a list of all audio files to play
    var stimSet = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Connect data
        self.stimulusSet.delegate = self
        self.stimulusSet.dataSource = self
        
        //Input data into the stimulus list
        stimulusSetList = ["tasks", "tasks_short"]
        
        stimSet = stimulusSet.selectedRow(inComponent: 0)
    }
    

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return stimulusSetList.count
    }
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        stimSet = stimulusSet.selectedRow(inComponent: 0)
        updateFileList()
        return stimulusSetList[row]
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.destination is ViewController {
            let vc = segue.destination as? ViewController
            vc?.stimSet = stimulusSet.selectedRow(inComponent: 0)
        }
    }
    
    func updateFileList() {
        fileList = [String]() // This will be a list of all audio files to play

        print("The chosen stimulus set is")
        print(stimSet+1)
        //TIMITsubset.text = String(format: "TIMIT%d", stimSet+1)
        let filename = "tasks"
        if let path = Bundle.main.path(forResource: filename, ofType: "txt") {
            do {
                let text = try String(contentsOfFile: path, encoding: String.Encoding.utf8)
                print(text)
                fileList = text.components(separatedBy: "\n")
                fileList = fileList.filter({ $0 != ""}) // Get rid of blanks
            } catch {
                print("Failed to read text from \(filename)")
            }
        } else {
            print("Failed to load file from app bundle \(filename)")
        }
    }
    
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

}
